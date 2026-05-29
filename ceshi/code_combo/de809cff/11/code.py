# Combo variant of de809cff (index 11)
# combo_spec: D1/3+D18/0
# Name: 颜色交换补丁+重叠区域teal标记冲突

from __future__ import annotations

# Design analysis for combo D1/3 + D18/0
# ---------------------------------------
# a) Concretization and behavior trace
#    - The cleaned scene still contains exactly two large foreground regions on black
#      background, plus true black dents and removable one-cell noise.
#    - D1/3 changes a dent from "paint the full 3x3 as the opposite color" into
#      "swap the two foreground colors wherever they already exist inside the 3x3";
#      black stays black and the center becomes teal.
#    - D18/0 changes the multi-dent composition rule: overlapping 3x3 action windows
#      are not resolved by order or parity anymore; any pixel touched by 2+ windows is
#      marked teal as a collision.
#    - Small local trace with clean neighborhood
#          A A B
#          A 0 B
#          A A B
#      Old task:
#          B B B
#          B 8 B
#          B B B
#      D1-only:
#          B B A
#          B 8 A
#          B B A
#      Combo with a second dent whose 3x3 overlaps this one:
#        pixels seen by only one dent still follow the D1-style swap, but every shared
#        overlap pixel becomes teal, so the overlap strip is neither A nor B.
#
# b) Complexity calibration
#    - If dents are all deep inside pure-color interiors, the swap rule collapses to the
#      old solid patch visually, so the generator must force many mixed dents whose 3x3
#      already contains both foreground colors.
#    - If all dent centers remain Chebyshev-separated by >= 3, there is no visible
#      overlap conflict, so the generator must also insert a few deliberate distance-2
#      pairs to create extra teal away from dent centers.
#    - Too many collisions would turn outputs into messy teal carpets; the generator
#      keeps only 1-2 collision pairs and preserves ordinary spacing elsewhere.
#
# c) Implicit assumptions audit
#    - `_main_colors` still assumes exactly two salient non-black colors; generation
#      keeps that invariant.
#    - `_clean_foreground` still assumes corruption is single-cell specks / protrusions;
#      the combo does not alter the noise model.
#    - `_dent_centers` still assumes true dents are original black cells whose cleaned
#      3x3 has a strict foreground majority. The generator blackens only cells that keep
#      that majority even when special collision pairs are punched together.
#    - The base writeback `out[r0:r1, c0:c1] = other` is invalid now in two ways:
#      1) single-window action is swap-on-foreground rather than unconditional repaint;
#      2) multi-window overlaps require explicit conflict marking instead of overwrite or
#         parity composition.
#    - Therefore the solver must accumulate cover counts first, then apply:
#         cover == 0 -> keep clean base
#         cover == 1 -> swap A<->B only on existing foreground
#         cover >= 2 -> teal conflict
#
# d) Placement legality
#    - Every dent center is an in-bounds foreground cell in the clean scene, and every
#      chosen collision pair is axis-aligned at distance 2 so their clipped 3x3 windows
#      overlap but the centers themselves do not touch.
#    - Collision pairs are only accepted if both dents remain detectable after both
#      centers are punched black simultaneously.
#    - All non-paired centers remain at Chebyshev distance >= 3 from every reserved
#      center, so collision behavior is localized and intentional.
#
# e) Visual observability
#    - Across 2-4 examples, a solver can notice two repeated signals:
#      1) mixed dents produce A/B mosaics instead of uniform opposite-color squares;
#      2) some overlap strips add extra teal pixels that are not dent centers.
#    - These cues are visible directly from geometry and local color context; nothing
#      depends on numeric color ids or hidden ordering.
#
# f) Visual cleanliness
#    - The palette remains black + two foreground colors + teal, so the third state is
#      easy to read.
#    - Extra teal is constrained to thin overlap bands near a few dent pairs, not broad
#      random blotches, and the rest of the dents stay spaced out.
#    - Mixed dents are common enough to expose swap behavior but not so dense that the
#      whole region boundary dissolves into checkerboard noise.
#
# g) Change inferability
#    - The two active mechanisms are structurally separable and both repeated: lone dents
#      show the local A<->B swap, while overlapping dents show conflict teal.
#    - The generator rejects samples whose combo output is too close to either the
#      D18-only solid-patch variant or the D1-only no-conflict variant, ensuring the
#      fusion behavior is actually learnable from examples.
#
# 9) New solver assumptions checked against the generator
#    - `cover >= 2` means intentional overlap, not accidental crowding; generation
#      creates collisions only via reserved distance-2 pairs and spaces all other dents.
#    - `cover == 1` swap logic assumes the starting pixel is either A, B, or black; the
#      cleaned base contains only those states by construction.
#    - `_dent_centers` scans original black cells, so every true event is blackened in
#      the input and any extra black noise is kept sparse enough to fail the majority
#      test after cleanup.

import random
import sys
from collections import Counter as _Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(host: int, colors: tuple[int, int]) -> int:
    return int(colors[1]) if int(host) == int(colors[0]) else int(colors[0])


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
    protruding wrong-color cells, the second lets their neighboring true cells
    keep the corrected local majority.
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
        other = _other_color(int(host), colors)
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _patch_coverages(
    shape: tuple[int, int], centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    cover = np.zeros((h, w), dtype=np.int16)
    unique_color = np.full((h, w), -1, dtype=np.int16)

    for r, c, host in centers:
        other = _other_color(host, colors)
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        view = unique_color[r0:r1, c0:c1]
        unset = view == -1
        view[unset] = int(other)
        unique_color[r0:r1, c0:c1] = view
        cover[r0:r1, c0:c1] += 1

    return cover, unique_color


def _compose_combo(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    """Apply D1 swaps on single-covered pixels and D18 teal on any overlap."""
    out = clean.copy()
    cover, _ = _patch_coverages(clean.shape, centers, colors)
    a, b = map(int, colors)

    lone = cover == 1
    conflict = cover >= 2
    out[lone & (clean == a)] = b
    out[lone & (clean == b)] = a
    out[conflict] = Color.TEAL

    for r, c, _host in centers:
        out[r, c] = Color.TEAL
    return out


def _compose_collision_only(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    """D18-only reference: solid opposite-color patch with teal overlap conflicts."""
    out = clean.copy()
    cover, unique_color = _patch_coverages(clean.shape, centers, colors)

    lone = cover == 1
    conflict = cover >= 2
    out[lone] = unique_color[lone]
    out[conflict] = Color.TEAL

    for r, c, _host in centers:
        out[r, c] = Color.TEAL
    return out


def _compose_swap_only(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    """D1-only reference: overlapping swaps compose by parity and never create teal conflicts."""
    h, w = clean.shape
    a, b = map(int, colors)
    toggles = np.zeros((h, w), dtype=np.uint8)
    for r, c, _host in centers:
        toggles[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] ^= 1

    out = clean.copy()
    odd = toggles.astype(bool)
    out[odd & (clean == a)] = b
    out[odd & (clean == b)] = a

    for r, c, _host in centers:
        out[r, c] = Color.TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into local color swaps with teal collision bands.

    Input structure:
    - Black background with two large foreground colors.
    - The foreground regions contain true black dents plus a few one-cell foreground
      specks or cross-boundary protrusions that should be cleaned away first.

    Rule:
    - First denoise the two foreground regions exactly as in the base task: isolated
      specks drop to black and single-cell boundary contamination is restored to its
      surrounding region.
    - Each original black dent whose cleaned 3x3 neighborhood is mostly one foreground
      color defines a clipped 3x3 action window.
    - Pixels covered by exactly one action window swap the two foreground colors if they
      were already foreground; black stays black. Pixels covered by two or more windows
      become teal (8) to mark a collision, and every dent center is also teal (8).

    Key insight: a true dent is no longer a solid opposite-color stamp; it toggles local
    foreground colors, and any overlap between two such toggle windows creates extra teal
    conflict marks beyond the original dent centers.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    return _compose_combo(clean, centers, colors)


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
COLLISION_PAIR_DISTANCE = 2
COLLISION_PAIR_COUNT_MIN, COLLISION_PAIR_COUNT_MAX = 1, 2
MIXED_DENT_OTHER_MIN = 1
MIXED_DENT_RATIO_MIN = 0.25

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: anti-degeneration ---
MIN_EXTRA_CONFLICT_TEAL = 3
COMBO_DIFF_VS_D18_MIN = 5
COMBO_DIFF_VS_D1_MIN = 3

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 800


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


def _dent_neighbor_counts(grid: np.ndarray, r: int, c: int, host: int, other: int) -> tuple[int, int]:
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    other_count = int(np.sum(sub == other))
    return host_count, other_count


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    host_count, other_count = _dent_neighbor_counts(grid, r, c, host, other)
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_count


def _is_mixed_dent(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    _host_count, other_count = _dent_neighbor_counts(grid, r, c, host, other)
    return other_count >= MIXED_DENT_OTHER_MIN


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


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    count: int,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    blocked_set = set(blocked)
    anchors = list(blocked)
    pool = candidates[:]
    rng.shuffle(pool)
    picked: list[tuple[int, int]] = []
    for r, c in pool:
        cell = (int(r), int(c))
        if cell in blocked_set:
            continue
        if all(_cheb(cell, other) >= DENT_PATCH_SEPARATION for other in anchors):
            picked.append(cell)
            blocked_set.add(cell)
            anchors.append(cell)
            if len(picked) == count:
                return picked
    return None


def _pick_visible_dents(
    all_candidates: list[tuple[int, int]],
    mixed_candidates: list[tuple[int, int]],
    count: int,
    blocked: list[tuple[int, int]],
    already_mixed: int,
    total_quota: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    need_total_mixed = max(1, int(np.ceil(total_quota * MIXED_DENT_RATIO_MIN)))
    min_more_mixed = max(0, need_total_mixed - already_mixed)
    max_more_mixed = min(count, len(mixed_candidates))
    for want_mixed in range(max_more_mixed, min_more_mixed - 1, -1):
        chosen_mixed = _pick_dent_centers(mixed_candidates, want_mixed, blocked, rng)
        if chosen_mixed is None:
            continue
        chosen_set = set(chosen_mixed)
        remaining_pool = [cell for cell in all_candidates if cell not in chosen_set]
        chosen_rest = _pick_dent_centers(remaining_pool, count - want_mixed, blocked + chosen_mixed, rng)
        if chosen_rest is not None:
            return chosen_mixed + chosen_rest
    return None


def _is_detectable_dent_set(
    clean: np.ndarray, dents: list[tuple[int, int]], host: int, other: int
) -> bool:
    temp = clean.copy()
    for r, c in dents:
        if int(temp[r, c]) != int(host):
            return False
        temp[r, c] = Color.BLACK
    for r, c in dents:
        sub = _sub3x3(temp, r, c)
        host_count = int(np.sum(sub == host))
        other_count = int(np.sum(sub == other))
        if host_count < DENT_MIN_HOST_3x3 or host_count <= other_count:
            return False
    return True


def _collision_pair_candidates(
    clean: np.ndarray,
    host: int,
    other: int,
    mixed_set: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    h, w = clean.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(clean[r, c]) != int(host) or not _is_safe_dent_center(clean, r, c, host, other):
                continue
            for dr, dc in ((0, COLLISION_PAIR_DISTANCE), (COLLISION_PAIR_DISTANCE, 0)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(clean[rr, cc]) != int(host) or not _is_safe_dent_center(clean, rr, cc, host, other):
                    continue
                pair = [(r, c), (rr, cc)]
                if not _is_detectable_dent_set(clean, pair, host, other):
                    continue
                mixed_score = int((r, c) in mixed_set) + int((rr, cc) in mixed_set)
                other_score = _dent_neighbor_counts(clean, r, c, host, other)[1] + _dent_neighbor_counts(clean, rr, cc, host, other)[1]
                candidates.append(((r, c), (rr, cc), mixed_score, other_score))
    return candidates


def _pick_collision_pairs(
    pair_candidates_by_color: dict[int, list[tuple[tuple[int, int], tuple[int, int], int, int]]],
    quota: dict[int, int],
    colors: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, tuple[tuple[int, int], tuple[int, int]], int, int]] | None:
    total_pair_budget = quota[int(colors[0])] // 2 + quota[int(colors[1])] // 2
    max_pairs = min(COLLISION_PAIR_COUNT_MAX, total_pair_budget)
    if max_pairs < COLLISION_PAIR_COUNT_MIN:
        return None

    pool: list[tuple[int, tuple[tuple[int, int], tuple[int, int]], int, int]] = []
    for color, entries in pair_candidates_by_color.items():
        local = entries[:]
        rng.shuffle(local)
        local.sort(key=lambda item: (item[2], item[3]), reverse=True)
        pool.extend((int(color), (item[0], item[1]), item[2], item[3]) for item in local)

    if not pool:
        return None

    for target in range(min(max_pairs, len(pool)), COLLISION_PAIR_COUNT_MIN - 1, -1):
        for _ in range(6):
            shuffled = pool[:]
            rng.shuffle(shuffled)
            shuffled.sort(key=lambda item: (item[2], item[3]), reverse=True)
            remaining = quota.copy()
            used: list[tuple[int, int]] = []
            chosen: list[tuple[int, tuple[tuple[int, int], tuple[int, int]], int, int]] = []
            for color, pair, mixed_score, other_score in shuffled:
                if len(chosen) >= target or remaining[color] < 2:
                    continue
                members = [pair[0], pair[1]]
                if any(_cheb(member, other) < DENT_PATCH_SEPARATION for member in members for other in used):
                    continue
                chosen.append((color, pair, mixed_score, other_score))
                remaining[color] -= 2
                used.extend(members)
                if len(chosen) == target:
                    return chosen
    return None


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
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


def _add_boundary_noise(inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random) -> None:
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


def _variant_metrics(
    clean: np.ndarray,
    centers: list[tuple[int, int, int]],
    colors: tuple[int, int],
) -> dict[str, int]:
    cover, _ = _patch_coverages(clean.shape, centers, colors)
    combo = _compose_combo(clean, centers, colors)
    collision_only = _compose_collision_only(clean, centers, colors)
    swap_only = _compose_swap_only(clean, centers, colors)

    center_mask = np.zeros(clean.shape, dtype=bool)
    mixed_centers = 0
    for r, c, host in centers:
        center_mask[r, c] = True
        if _is_mixed_dent(clean, r, c, host, _other_color(host, colors)):
            mixed_centers += 1

    conflict = cover >= 2
    return {
        "dent_cells": len(centers),
        "mixed_centers": mixed_centers,
        "conflict_pixels": int(np.sum(conflict)),
        "extra_conflict_teal": int(np.sum(conflict & ~center_mask)),
        "max_cover": int(np.max(cover)),
        "diff_vs_d18": int(np.count_nonzero(combo != collision_only)),
        "diff_vs_d1": int(np.count_nonzero(combo != swap_only)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    quota = {int(color_a): n_total // 2, int(color_b): n_total - n_total // 2}
    if rng.random() < 0.5:
        quota[int(color_a)], quota[int(color_b)] = quota[int(color_b)], quota[int(color_a)]

    h, w = clean.shape
    valid = {int(color_a): [], int(color_b): []}
    mixed = {int(color_a): [], int(color_b): []}
    mixed_sets = {int(color_a): set(), int(color_b): set()}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in colors:
                continue
            other = _other_color(v, colors)
            if not _is_safe_dent_center(clean, r, c, v, other):
                continue
            valid[v].append((r, c))
            if _is_mixed_dent(clean, r, c, v, other):
                mixed[v].append((r, c))
                mixed_sets[v].add((r, c))

    if len(valid[int(color_a)]) < quota[int(color_a)] + 2 or len(valid[int(color_b)]) < quota[int(color_b)] + 2:
        return None

    pair_candidates = {
        int(color_a): _collision_pair_candidates(clean, int(color_a), int(color_b), mixed_sets[int(color_a)]),
        int(color_b): _collision_pair_candidates(clean, int(color_b), int(color_a), mixed_sets[int(color_b)]),
    }
    if not pair_candidates[int(color_a)] and not pair_candidates[int(color_b)]:
        return None

    collision_pairs = _pick_collision_pairs(pair_candidates, quota.copy(), colors, rng)
    if collision_pairs is None:
        return None

    chosen_by_color = {int(color_a): [], int(color_b): []}
    blocked_pairs: list[tuple[int, int]] = []
    pre_mixed = {int(color_a): 0, int(color_b): 0}
    remaining = quota.copy()
    for host, pair, mixed_score, _other_score in collision_pairs:
        chosen_by_color[host].extend([pair[0], pair[1]])
        blocked_pairs.extend([pair[0], pair[1]])
        pre_mixed[host] += mixed_score
        remaining[host] -= 2

    selection: dict[int, list[tuple[int, int]]] | None = None
    orders = [list(colors), [int(color_b), int(color_a)]]
    if rng.random() < 0.5:
        orders.reverse()
    for order in orders:
        local = {int(color_a): chosen_by_color[int(color_a)][:], int(color_b): chosen_by_color[int(color_b)][:]}
        blocked_now = blocked_pairs[:]
        ok = True
        for host in order:
            picks = _pick_visible_dents(
                valid[host],
                mixed[host],
                remaining[host],
                blocked_now,
                pre_mixed[host],
                quota[host],
                rng,
            )
            if picks is None:
                ok = False
                break
            local[host].extend(picks)
            blocked_now.extend(picks)
        if ok:
            selection = local
            break
    if selection is None:
        return None

    centers: list[tuple[int, int, int]] = []
    for host in colors:
        centers.extend((r, c, int(host)) for r, c in selection[int(host)])

    metrics = _variant_metrics(clean, centers, colors)
    if metrics["extra_conflict_teal"] < MIN_EXTRA_CONFLICT_TEAL:
        return None
    if metrics["diff_vs_d18"] < COMBO_DIFF_VS_D18_MIN or metrics["diff_vs_d1"] < COMBO_DIFF_VS_D1_MIN:
        return None
    if metrics["mixed_centers"] < max(2, int(np.ceil(len(centers) * MIXED_DENT_RATIO_MIN))):
        return None
    if metrics["max_cover"] < 2:
        return None

    output = _compose_combo(clean, centers, colors)

    inp = clean.copy()
    for r, c, _host in centers:
        inp[r, c] = Color.BLACK

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

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
