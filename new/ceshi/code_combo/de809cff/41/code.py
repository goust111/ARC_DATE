# Combo variant of de809cff (index 41)
# combo_spec: D1/3+D16/2
# Name: 补丁镜像复制+颜色交换补丁

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
#   - Clean scene shape stays close to the source task: black background, two large
#     foreground regions, black dents, and a few one-cell noise specks.
#   - The original task turned a dent inside host color A into a solid 3x3 patch of
#     color B with a teal center.
#   - D1/3 changes that local action into a bidirectional swap on foreground cells.
#     Example local clean 3x3:
#         A A B
#         A 0 B
#         A A B
#     old output:
#         B B B
#         B 8 B
#         B B B
#     new local output:
#         B B A
#         B 8 A
#         B B A
#   - D16/2 then adds a second action center at the point-reflected location
#     (H-1-r, W-1-c). That remote 3x3 gets the same kind of swap treatment and its
#     center is also marked teal. So one real dent produces two visible swap patches:
#     one local, one remote.
# b) complexity_calibration:
#   - If dents were only placed deep inside pure-color interiors, swap vs solid-paint
#     would be hard to notice. The generator therefore prefers dents whose local or
#     mirrored 3x3 already contains a few pixels of the opposite foreground color.
#   - Mirror duplication doubles the number of visible patches, so the underlying dent
#     count is reduced slightly and pair-dents are disabled to keep samples readable.
# c) implicit_assumption_review:
#   - `_main_colors` still assumes exactly two non-black foreground colors; the
#     generator keeps that invariant.
#   - `_clean_foreground` still assumes noise is sparse one-cell corruption. The
#     fusion does not alter the noise model, so cleanup logic remains valid.
#   - `_dent_centers` only looks at original black cells. This remains correct because
#     mirrored patch centers are not black in the input; they are inferred in output.
#   - The new action stage assumes the mirrored center should lie in the opposite
#     foreground region and be far enough from other effect centers to remain visible.
#     `generate_input()` guarantees both.
#   - Overlapping swap windows would introduce ordering ambiguity if processed one by
#     one. The solver therefore accumulates all local+mirrored 3x3 coverages and
#     toggles by parity in one shot.
#   - Teal must be written after all swaps; otherwise a later toggle could overwrite
#     the remote marker.
# d) placement_legality:
#   - Every chosen dent center is a valid source-task dent in its host region.
#   - Its point reflection is required to lie in the opposite region and to have a
#     clean 3x3 majority for that opposite color.
#   - Selection uses spacing over the full set of effect centers (originals and
#     mirrors), so 3x3 patches do not overlap or collapse into ambiguous blobs.
# e) visual_observability:
#   - A human can first detect the same black dents as in the source task, then notice
#     that each dent creates a second teal-centered patch on the opposite side of the
#     board by point symmetry.
#   - Within both the local and remote patches, mixed neighborhoods reveal that the
#     action is a color swap, not a solid repaint.
# f) cleanliness:
#   - Only two foreground colors plus teal are used in the output, matching the source
#     task's visual vocabulary.
#   - Remote effects are kept separated and centered inside foreground regions, so the
#     output reads as two families of 3x3 patch events instead of a noisy flood-fill.
# g) inferability_of_color_change:
#   - The color rule remains structured: only the two region colors participate, and
#     only inside 3x3 windows centered on real or mirrored dent centers.
#   - Because the same A<->B swap appears both locally and remotely across repeated
#     examples, the solver can infer a consistent two-color toggle rather than an
#     arbitrary recoloring lookup.
# h) new_solver_assumptions_checked_against_generator:
#   - `main()` assumes point reflection across the whole grid, so the generator avoids
#     self-mirroring near the exact grid center.
#   - `main()` marks every effect center teal; the generator ensures those remote
#     centers land on meaningful foreground cells, not on empty background.


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
    protruding wrong-color cells, the second lets neighboring true cells keep the
    corrected local majority.
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


def _mirror_point(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    return int(h - 1 - r), int(w - 1 - c)


def _effect_centers_from_dents(
    dents: list[tuple[int, int, int]], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    ordered: dict[tuple[int, int], None] = {}
    for r, c, _host in dents:
        ordered[(int(r), int(c))] = None
        ordered[_mirror_point(shape, int(r), int(c))] = None
    return list(ordered.keys())


def _apply_swap_effects(
    clean: np.ndarray, effect_centers: list[tuple[int, int]], colors: tuple[int, int]
) -> np.ndarray:
    """Toggle the two foreground colors in every 3x3 window centered on an effect point."""
    h, w = clean.shape
    a, b = map(int, colors)
    cover = np.zeros((h, w), dtype=np.uint8)
    for r, c in effect_centers:
        cover[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] ^= 1

    out = clean.copy()
    odd = cover.astype(bool)
    mask_a = odd & (clean == a)
    mask_b = odd & (clean == b)
    out[mask_a] = b
    out[mask_b] = a

    for r, c in effect_centers:
        out[r, c] = Color.TEAL
    return out


def _apply_solid_mirror_patches(
    clean: np.ndarray, dents: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    """Baseline used only for anti-degeneration checks inside the generator."""
    h, w = clean.shape
    out = clean.copy()
    for r, c, host in dents:
        stamp = colors[1] if int(host) == int(colors[0]) else colors[0]
        for rr, cc in ((int(r), int(c)), _mirror_point(clean.shape, int(r), int(c))):
            out[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)] = int(stamp)
    for rr, cc in _effect_centers_from_dents(dents, clean.shape):
        out[rr, cc] = Color.TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents into paired local-and-mirrored color-swap patches.

    Input structure:
    - Black background with two large foreground colors.
    - The foreground regions contain true black dents plus sparse one-cell specks
      or short protrusions near the boundary between the two regions.

    Rule:
    - First clean the two regions exactly as in the source task: isolated specks
      revert to black, and one-cell cross-boundary contamination is corrected.
    - Each original black dent whose cleaned 3x3 neighborhood is mostly one
      foreground color creates a local 3x3 A<->B swap window at that dent center.
    - The same dent also creates a second swap window at the point-reflected
      position across the whole grid. All effect centers, both local and mirrored,
      are finally colored teal (8).

    Key insight: Only the original black holes reveal where the process starts, but
    every valid dent broadcasts its swap patch to a symmetric remote partner. The
    output is therefore the cleaned grid after applying one shared family of local
    and mirrored foreground-color toggles.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    dents = _dent_centers(g, clean, colors, bg)
    effect_centers = _effect_centers_from_dents(dents, g.shape)
    return _apply_swap_effects(clean, effect_centers, colors)


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 8, 12
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
MIRROR_SELF_SEPARATION_MIN = 6
MIXED_DENT_OTHER_MIN = 1
MIXED_DENT_RATIO_MIN = 0.25
PAIR_PICK_ATTEMPTS = 50

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: anti-degeneration ---
REMOTE_SWAP_DIFF_PIXELS_MIN = 8
SWAP_VS_SOLID_DIFF_PIXELS_MIN = 8
REMOTE_NON_TEAL_DIFF_PIXELS_MIN = 4

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 500


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Build two large point-reflected rectangles so mirrored dents land in the opposite region."""
    size_min = max(int(np.ceil(N * max(RECT_SIZE_FRAC_MIN, 0.56))), (N // 2) + 1)
    size_max = min(int(np.floor(N * RECT_SIZE_FRAC_MAX)), N - 1)
    if size_min > size_max:
        return None

    for _ in range(40):
        hh = rng.randint(size_min, size_max)
        ww = rng.randint(size_min, size_max)
        max_r0 = min(RECT_MARGIN_MAX, N - hh)
        max_c0 = min(RECT_MARGIN_MAX, N - ww)
        r0 = rng.randint(0, max_r0)
        c0 = rng.randint(0, max_c0)
        r1 = N - hh - r0
        c1 = N - ww - c0
        if r1 < 0 or c1 < 0:
            continue

        mask_a = np.zeros((N, N), dtype=bool)
        mask_b = np.zeros((N, N), dtype=bool)
        mask_a[r0 : r0 + hh, c0 : c0 + ww] = True
        mask_b[r1 : r1 + hh, c1 : c1 + ww] = True
        overlap = mask_a & mask_b
        overlap_rows = int(np.sum(np.any(overlap, axis=1)))
        overlap_cols = int(np.sum(np.any(overlap, axis=0)))
        overlap_count = int(np.sum(overlap))
        if overlap_rows < 3 or overlap_cols < 3:
            continue

        grid = np.zeros((N, N), dtype=int)
        grid[mask_a] = color_a
        grid[mask_b] = color_b

        cnt_a = int(np.sum(grid == color_a))
        cnt_b = int(np.sum(grid == color_b))
        coverage = (cnt_a + cnt_b) / (N * N)
        if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
            continue
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            continue
        if overlap_count < 25 or overlap_count > int(0.55 * hh * ww):
            continue
        return grid

    return None


def _dent_neighbor_counts(grid: np.ndarray, r: int, c: int, host: int, other: int) -> tuple[int, int]:
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    other_count = int(np.sum(sub == other))
    return host_count, other_count


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """After setting (r,c) to black, the solver must detect it as a dent in host region."""
    if int(grid[r, c]) != int(host):
        return False
    host_count, other_count = _dent_neighbor_counts(grid, r, c, host, other)
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_count


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


def _far_from(r: int, c: int, points: list[tuple[int, int]], min_sep: int) -> bool:
    return all(max(abs(int(r) - rr), abs(int(c) - cc)) >= min_sep for rr, cc in points)


def _candidate_from_cell(
    clean: np.ndarray, r: int, c: int, host: int, other: int
) -> dict[str, int | bool | tuple[int, int]] | None:
    if not _is_safe_dent_center(clean, r, c, host, other):
        return None
    mr, mc = _mirror_point(clean.shape, r, c)
    if max(abs(int(r) - int(mr)), abs(int(c) - int(mc))) < MIRROR_SELF_SEPARATION_MIN:
        return None
    if not _is_safe_dent_center(clean, mr, mc, other, host):
        return None

    local_other = _dent_neighbor_counts(clean, r, c, host, other)[1]
    remote_other = _dent_neighbor_counts(clean, mr, mc, other, host)[1]
    return {
        "r": int(r),
        "c": int(c),
        "host": int(host),
        "mirror": (int(mr), int(mc)),
        "mixed": bool(local_other >= MIXED_DENT_OTHER_MIN or remote_other >= MIXED_DENT_OTHER_MIN),
        "score": int(local_other + remote_other),
    }


def _candidate_effect_points(candidate: dict[str, int | bool | tuple[int, int]]) -> list[tuple[int, int]]:
    mr, mc = candidate["mirror"]
    return [(int(candidate["r"]), int(candidate["c"])), (int(mr), int(mc))]


def _can_place_candidate(
    candidate: dict[str, int | bool | tuple[int, int]],
    blocked_effects: list[tuple[int, int]],
) -> bool:
    pts = _candidate_effect_points(candidate)
    for r, c in pts:
        if not _far_from(r, c, blocked_effects, DENT_PATCH_SEPARATION):
            return False
    return True


def _greedy_pick_candidates(
    candidates: list[dict[str, int | bool | tuple[int, int]]],
    n: int,
    blocked_effects: list[tuple[int, int]],
    rng: random.Random,
) -> list[dict[str, int | bool | tuple[int, int]]] | None:
    if n == 0:
        return []
    for _ in range(PAIR_PICK_ATTEMPTS):
        pool = candidates.copy()
        rng.shuffle(pool)
        pool.sort(key=lambda cand: (int(cand["score"]), bool(cand["mixed"])), reverse=True)
        chosen: list[dict[str, int | bool | tuple[int, int]]] = []
        used = list(blocked_effects)
        for cand in pool:
            if not _can_place_candidate(cand, used):
                continue
            chosen.append(cand)
            used.extend(_candidate_effect_points(cand))
            if len(chosen) >= n:
                return chosen
    return None


def _pick_visible_candidates(
    candidates: list[dict[str, int | bool | tuple[int, int]]],
    n: int,
    blocked_effects: list[tuple[int, int]],
    rng: random.Random,
) -> list[dict[str, int | bool | tuple[int, int]]] | None:
    mixed = [cand for cand in candidates if bool(cand["mixed"])]
    plain = [cand for cand in candidates if not bool(cand["mixed"])]
    min_mixed = min(len(mixed), max(1, int(np.ceil(n * MIXED_DENT_RATIO_MIN))))
    max_mixed = min(n, len(mixed))

    if max_mixed == 0:
        return _greedy_pick_candidates(candidates, n, blocked_effects, rng)

    for want_mixed in range(max_mixed, min_mixed - 1, -1):
        chosen_mixed = _greedy_pick_candidates(mixed, want_mixed, blocked_effects, rng)
        if chosen_mixed is None:
            continue
        chosen_keys = {(int(cand["r"]), int(cand["c"])) for cand in chosen_mixed}
        rest_pool = [cand for cand in plain + mixed if (int(cand["r"]), int(cand["c"])) not in chosen_keys]
        extra_blocked = list(blocked_effects)
        for cand in chosen_mixed:
            extra_blocked.extend(_candidate_effect_points(cand))
        chosen_rest = _greedy_pick_candidates(rest_pool, n - want_mixed, extra_blocked, rng)
        if chosen_rest is not None:
            return chosen_mixed + chosen_rest
    return None


def _dents_from_candidates(
    candidates: list[dict[str, int | bool | tuple[int, int]]]
) -> list[tuple[int, int, int]]:
    return [(int(cand["r"]), int(cand["c"]), int(cand["host"])) for cand in candidates]


def _blacken_dents(clean: np.ndarray, dents: list[tuple[int, int, int]]) -> np.ndarray:
    out = clean.copy()
    for r, c, _host in dents:
        out[int(r), int(c)] = Color.BLACK
    return out


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


def _local_only_swap_output(
    clean: np.ndarray, dents: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    return _apply_swap_effects(clean, [(int(r), int(c)) for r, c, _host in dents], colors)


def _remote_swap_signal(
    output: np.ndarray, local_only: np.ndarray
) -> tuple[int, int]:
    diff = output != local_only
    remote_total = int(np.count_nonzero(diff))
    remote_non_teal = int(np.count_nonzero(diff & (output != Color.TEAL) & (local_only != Color.TEAL)))
    return remote_total, remote_non_teal


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
    candidates = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a:
                cand = _candidate_from_cell(clean, r, c, color_a, color_b)
                if cand is not None:
                    candidates[color_a].append(cand)
            elif v == color_b:
                cand = _candidate_from_cell(clean, r, c, color_b, color_a)
                if cand is not None:
                    candidates[color_b].append(cand)

    if len(candidates[color_a]) < n_a or len(candidates[color_b]) < n_b:
        return None

    dents: list[tuple[int, int, int]] | None = None
    orderings = [(color_a, color_b), (color_b, color_a)]
    if rng.random() < 0.5:
        orderings.reverse()
    for first_color, second_color in orderings:
        first_need = n_a if first_color == color_a else n_b
        second_need = n_b if first_color == color_a else n_a

        chosen_first = _pick_visible_candidates(candidates[first_color], first_need, [], rng)
        if chosen_first is None:
            continue
        blocked_effects: list[tuple[int, int]] = []
        for cand in chosen_first:
            blocked_effects.extend(_candidate_effect_points(cand))

        chosen_second = _pick_visible_candidates(candidates[second_color], second_need, blocked_effects, rng)
        if chosen_second is None:
            continue

        if first_color == color_a:
            dents = _dents_from_candidates(chosen_first + chosen_second)
        else:
            dents = _dents_from_candidates(chosen_second + chosen_first)
        break

    if dents is None:
        return None

    colors = (int(color_a), int(color_b))
    output = _apply_swap_effects(clean, _effect_centers_from_dents(dents, clean.shape), colors)
    local_only = _local_only_swap_output(clean, dents, colors)
    solid_mirror = _apply_solid_mirror_patches(clean, dents, colors)

    remote_total, remote_non_teal = _remote_swap_signal(output, local_only)
    solid_diff = int(np.count_nonzero(output != solid_mirror))
    if remote_total < REMOTE_SWAP_DIFF_PIXELS_MIN:
        return None
    if remote_non_teal < REMOTE_NON_TEAL_DIFF_PIXELS_MIN:
        return None
    if solid_diff < SWAP_VS_SOLID_DIFF_PIXELS_MIN:
        return None

    inp = _blacken_dents(clean, dents)

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
