# Combo variant of de809cff (index 30)
# combo_spec: D4/0+D5/2+D9/0
# Name: 三色区域+同心嵌套+补丁裁剪

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - Base task: two large colors occupy most of the board; a black dent inside
#      one color becomes a full 3x3 patch of the only other color.
#    - Combo task: the clean scene is a three-layer nested structure:
#        outer ring / middle ring / inner core, each in a different color.
#      After cleanup, dents are still detected from black cells whose cleaned
#      3x3 neighborhood has a unique host-color majority.
#    - New mapping: patch color follows nesting depth, not "the only other":
#        outer -> middle, middle -> inner, inner -> outer.
#    - New write-back: the 3x3 patch is first proposed, then clipped so only
#      cells still inside the dent's own host region are recolored.
#    - Small trace:
#        outer dent near the canvas-facing boundary used to stamp a full 3x3
#        block that leaked into black background; now only outer-ring cells flip.
#        middle dents can sit against either the outer-facing boundary or the
#        inner-facing boundary of the ring; only the middle-ring subset flips,
#        so samples visibly show clipping on both kinds of side.
#        inner dent uses the outer color, but its patch is heavily clipped by
#        the small inner rectangle boundary.
# b) Complexity calibration:
#    - The color rule still has only three stable cases because depth roles are
#      fixed and repeated in every sample.
#    - The generator forces dents in all three layers, plus several middle-layer
#      dents near the outer-facing side and several near the inner-facing side,
#      so clipping on both boundaries is visually repeated across each sample.
#    - The innermost region is kept small enough that clipping is common, while
#      the outer ring still provides some larger marks to contrast against.
# c) Hidden assumptions audited:
#    - `_main_colors` can no longer stop at two foreground colors.
#    - Cleanup can no longer compare a cell against one rival color; it needs a
#      unique dominant rival among three colors.
#    - Patch color cannot be "the other color"; it depends on inferred nesting
#      depth, so role inference must happen before stamping.
#    - Directly writing the whole 3x3 neighborhood is invalid because patches
#      must stay inside the reconstructed host-region mask.
#    - Role inference from the cleaned grid alone would be brittle if dents left
#      black cells on region borders; the solver therefore fills dent centers
#      back into a temporary region mask before inspecting adjacencies.
# d) Placement legality:
#    - The generator builds three strictly nested rectangles, then uses painter's
#      algorithm to expose them as an outer frame, a middle frame, and an inner
#      core. All bounds are checked before acceptance.
#    - Dent centers are chosen only from in-bounds host cells whose host color
#      remains the unique 3x3 majority after blackening the center.
#    - Noise is inserted only when local cleanup restores the original clean
#      scene, and every sample is validated by `main()` afterward.
# e) Visual observability:
#    - Depth roles are visible from containment alone; no rule depends on color
#      ids, frequency order, or hidden metadata.
#    - Each sample contains dents in all three regions, so the outer/middle/inner
#      color cycle is inferable from repeated paired examples.
#    - Middle-layer dents are explicitly forced to appear against both of the
#      ring's boundaries, making the two clipping directions easy to spot.
# f) Visual cleanliness:
#    - Dent centers remain spaced by Chebyshev distance >= 3 before optional
#      pair expansion, so clipped patches do not collapse into clutter.
#    - Only one overlay motif is used everywhere: recolored local patch + teal
#      center. Clipping removes spillover instead of adding extra markers.
# g) Change inferability:
#    - Color change is governed by one structural signal: nesting depth.
#    - Patch shape change is governed by one structural signal: whether the 3x3
#      box leaves the host region. Because both signals repeat in the same three
#      depth roles, the fused rule is still learnable from a few examples.


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == int(c))), reverse=True)
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
    winners = [int(color) for color, count in counts.items() if int(count) == int(best)]
    if best < threshold or len(winners) != 1:
        return None
    return int(winners[0])


def _clean_cell(cur: np.ndarray, r: int, c: int, colors: tuple[int, int, int], bg: int) -> int:
    v = int(cur[r, c])
    if v not in colors:
        return int(bg)

    if _has_same_2x2(cur, r, c, v):
        return int(v)

    sub = _sub3x3(cur, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    own = int(counts[int(v)])
    bg_count = int(np.sum(sub == int(bg)))
    dominant = _dominant_color(sub, colors, threshold=5)

    if dominant is not None and int(dominant) != int(v) and int(counts[int(dominant)]) > own:
        return int(dominant)
    if bg_count >= 5 and own <= 4:
        return int(bg)
    if own <= 2:
        return int(bg)
    return int(v)


def _clean_foreground(g: np.ndarray, colors: tuple[int, int, int], bg: int) -> np.ndarray:
    cur = np.asarray(g, dtype=int).copy()
    h, w = cur.shape

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                nxt[r, c] = _clean_cell(cur, r, c, colors, bg)
        cur = nxt
    return cur


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


def _reconstruct_regions(clean: np.ndarray, centers: list[tuple[int, int, int]]) -> np.ndarray:
    region_mask = clean.copy()
    for r, c, host in centers:
        region_mask[r, c] = int(host)
    return region_mask


def _adjacency_sets(region_mask: np.ndarray, colors: tuple[int, int, int], bg: int) -> dict[int, set[int]]:
    h, w = region_mask.shape
    adj = {int(color): set() for color in colors}
    for r in range(h):
        for c in range(w):
            host = int(region_mask[r, c])
            if host not in adj:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    other = int(region_mask[rr, cc])
                    if other != host:
                        adj[host].add(int(other))
    return adj


def _nesting_roles(region_mask: np.ndarray, colors: tuple[int, int, int], bg: int) -> dict[str, int]:
    adj = _adjacency_sets(region_mask, colors, bg)
    fg = {int(color) for color in colors}

    outer_candidates = [
        int(color)
        for color in colors
        if int(bg) in adj[int(color)] and len(adj[int(color)] & fg) == 1
    ]
    inner_candidates = [
        int(color)
        for color in colors
        if int(bg) not in adj[int(color)] and len(adj[int(color)] & fg) == 1
    ]
    assert len(outer_candidates) == 1, "expected unique outer ring"
    assert len(inner_candidates) == 1, "expected unique inner core"

    outer = int(outer_candidates[0])
    inner = int(inner_candidates[0])
    middle_candidates = [int(color) for color in colors if int(color) not in (outer, inner)]
    assert len(middle_candidates) == 1, "expected unique middle ring"
    middle = int(middle_candidates[0])
    assert adj[int(middle)] & fg == {int(outer), int(inner)}, "middle ring must touch both neighbors"
    return {"outer": int(outer), "middle": int(middle), "inner": int(inner)}


def _stamp_map(roles: dict[str, int]) -> dict[int, int]:
    return {
        int(roles["outer"]): int(roles["middle"]),
        int(roles["middle"]): int(roles["inner"]),
        int(roles["inner"]): int(roles["outer"]),
    }


def _paint_clipped_patch(
    out: np.ndarray, region_mask: np.ndarray, r: int, c: int, host: int, stamp_color: int
) -> None:
    h, w = out.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if int(region_mask[rr, cc]) == int(host):
                out[rr, cc] = int(stamp_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in three nested colored regions into clipped depth-cycle patches.

    Input structure:
    - Black background plus exactly three large foreground regions arranged as a
      nested outer frame, middle frame, and inner core.
    - The regions contain black dents and a few one-cell foreground specks or
      boundary intrusions that must be cleaned before the dents are interpreted.

    Rule:
    - First run the same local cleanup idea as the source task, but in a
      three-color setting: isolated foreground specks revert to black, and a
      lone cross-boundary intrusion flips back to the unique local foreground
      majority.
    - Each original black cell whose cleaned 3x3 neighborhood is uniquely
      dominated by one host region becomes a dent center. Reconstruct the host
      regions, infer which colors are outer/middle/inner, and stamp a 3x3 patch
      using the depth cycle outer -> middle -> inner -> outer.
    - Keep only the part of that patch that lies inside the dent's own host
      region, then color the dent center teal (8).

    Key insight: The solver must combine three cues at once: cleanup reveals the
    host region, nesting depth determines the replacement color, and the visible
    patch shape comes from clipping the 3x3 write-back to that reconstructed host.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    centers = _dent_centers(g, clean, colors, bg)
    region_mask = _reconstruct_regions(clean, centers)
    roles = _nesting_roles(region_mask, colors, bg)
    stamp_map = _stamp_map(roles)

    out = clean.copy()
    for r, c, host in centers:
        _paint_clipped_patch(out, region_mask, int(r), int(c), int(host), int(stamp_map[int(host)]))

    for r, c, _host in centers:
        out[int(r), int(c)] = int(Color.TEAL)
    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 24, 30
OUTER_MARGIN_MIN, OUTER_MARGIN_MAX = 1, 3
OUTER_SIZE_FRAC_MIN, OUTER_SIZE_FRAC_MAX = 0.74, 0.88
OUTER_RING_THICKNESS = (3, 6)
MIDDLE_RING_THICKNESS = (2, 4)
MIN_OUTER_CELLS = 120
MIN_MIDDLE_CELLS = 80
MIN_INNER_CELLS = 28
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84
BUILD_ATTEMPTS = 80

# --- CFG: dent placement ---
OUTER_DENTS_RANGE = (4, 5)
MIDDLE_DENTS_RANGE = (4, 5)
INNER_DENTS_RANGE = (4, 6)
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.08
SELECT_RETRY_ATTEMPTS = 180
MIN_OUTER_CLIPPED = 2
MIN_OUTER_INTERIOR = 1
MIN_MIDDLE_OUTER_TOUCH = 2
MIN_MIDDLE_INNER_TOUCH = 2
MIN_MIDDLE_CLIPPED = 3
MIN_INNER_CLIPPED_FRAC = 0.75
MIN_TOTAL_CLIP_LOSS = 16

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 8, 16
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.60, 0.80
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6
MIN_NOISE_PLACED = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 500


def _region_counts(grid: np.ndarray, roles: dict[str, int]) -> dict[str, int]:
    return {name: int(np.sum(grid == int(color))) for name, color in roles.items()}


def _build_regions(N: int, role_colors: dict[str, int], rng: random.Random) -> np.ndarray | None:
    outer = int(role_colors["outer"])
    middle = int(role_colors["middle"])
    inner = int(role_colors["inner"])

    for _ in range(BUILD_ATTEMPTS):
        outer_h = rng.randint(int(round(N * OUTER_SIZE_FRAC_MIN)), int(round(N * OUTER_SIZE_FRAC_MAX)))
        outer_w = rng.randint(int(round(N * OUTER_SIZE_FRAC_MIN)), int(round(N * OUTER_SIZE_FRAC_MAX)))
        outer_r0 = rng.randint(OUTER_MARGIN_MIN, max(OUTER_MARGIN_MIN, N - outer_h - OUTER_MARGIN_MIN))
        outer_c0 = rng.randint(OUTER_MARGIN_MIN, max(OUTER_MARGIN_MIN, N - outer_w - OUTER_MARGIN_MIN))
        outer_r1 = outer_r0 + outer_h
        outer_c1 = outer_c0 + outer_w

        top_outer = rng.randint(*OUTER_RING_THICKNESS)
        bottom_outer = rng.randint(*OUTER_RING_THICKNESS)
        left_outer = rng.randint(*OUTER_RING_THICKNESS)
        right_outer = rng.randint(*OUTER_RING_THICKNESS)
        middle_r0 = outer_r0 + top_outer
        middle_c0 = outer_c0 + left_outer
        middle_r1 = outer_r1 - bottom_outer
        middle_c1 = outer_c1 - right_outer
        middle_h = middle_r1 - middle_r0
        middle_w = middle_c1 - middle_c0
        if middle_h < 10 or middle_w < 10:
            continue

        top_middle = rng.randint(*MIDDLE_RING_THICKNESS)
        bottom_middle = rng.randint(*MIDDLE_RING_THICKNESS)
        left_middle = rng.randint(*MIDDLE_RING_THICKNESS)
        right_middle = rng.randint(*MIDDLE_RING_THICKNESS)
        inner_r0 = middle_r0 + top_middle
        inner_c0 = middle_c0 + left_middle
        inner_r1 = middle_r1 - bottom_middle
        inner_c1 = middle_c1 - right_middle
        inner_h = inner_r1 - inner_r0
        inner_w = inner_c1 - inner_c0
        if inner_h < 5 or inner_w < 5:
            continue

        grid = np.zeros((N, N), dtype=int)
        grid[outer_r0:outer_r1, outer_c0:outer_c1] = int(outer)
        grid[middle_r0:middle_r1, middle_c0:middle_c1] = int(middle)
        grid[inner_r0:inner_r1, inner_c0:inner_c1] = int(inner)

        coverage = float(np.mean(grid != int(Color.BLACK)))
        counts = _region_counts(grid, role_colors)
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            continue
        if counts["outer"] < MIN_OUTER_CELLS or counts["middle"] < MIN_MIDDLE_CELLS or counts["inner"] < MIN_INNER_CELLS:
            continue

        colors = (int(outer), int(middle), int(inner))
        if not np.array_equal(_clean_foreground(grid, colors, Color.BLACK), grid):
            continue
        if _nesting_roles(grid, colors, Color.BLACK) != role_colors:
            continue
        return grid
    return None


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, colors: tuple[int, int, int]) -> bool:
    h, w = grid.shape
    if not (1 <= int(r) < h - 1 and 1 <= int(c) < w - 1):
        return False
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, int(r), int(c))
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    counts[int(host)] -= 1
    host_count = int(counts[int(host)])
    other_max = max(int(counts[int(color)]) for color in colors if int(color) != int(host))
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_max


def _clip_loss(grid: np.ndarray, r: int, c: int, host: int) -> int:
    sub = _sub3x3(grid, int(r), int(c))
    return int(sub.size - np.sum(sub == int(host)))


def _middle_touches_outer(grid: np.ndarray, r: int, c: int, outer: int) -> bool:
    sub = _sub3x3(grid, int(r), int(c))
    return bool(np.any(sub == int(outer)))


def _middle_touches_inner(grid: np.ndarray, r: int, c: int, inner: int) -> bool:
    sub = _sub3x3(grid, int(r), int(c))
    return bool(np.any(sub == int(inner)))


def _pick_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    occupied: set[tuple[int, int]],
    *,
    preferred: list[tuple[int, int]] | None = None,
    min_preferred: int = 0,
) -> list[tuple[int, int]] | None:
    preferred = [] if preferred is None else [tuple(map(int, p)) for p in preferred if tuple(map(int, p)) in set(candidates)]
    base_candidates = [tuple(map(int, p)) for p in candidates]

    for _ in range(SELECT_RETRY_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        pref_count = 0

        pref_order = preferred[:]
        rng.shuffle(pref_order)
        for r, c in pref_order:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in occupied | set(chosen)):
                chosen.append((int(r), int(c)))
                pref_count += 1
                if pref_count >= min_preferred:
                    break
        if pref_count < min_preferred:
            continue

        remaining = [p for p in base_candidates if p not in chosen]
        rng.shuffle(remaining)
        for r, c in remaining:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in occupied | set(chosen)):
                chosen.append((int(r), int(c)))
                if len(chosen) >= n:
                    return chosen
    return None


def _pick_middle_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    occupied: set[tuple[int, int]],
    outer_touch: list[tuple[int, int]],
    inner_touch: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidate_set = set(tuple(map(int, p)) for p in candidates)
    outer_touch = [tuple(map(int, p)) for p in outer_touch if tuple(map(int, p)) in candidate_set]
    inner_touch = [tuple(map(int, p)) for p in inner_touch if tuple(map(int, p)) in candidate_set]

    for _ in range(SELECT_RETRY_ATTEMPTS):
        chosen: list[tuple[int, int]] = []

        pool = outer_touch[:]
        rng.shuffle(pool)
        for r, c in pool:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in occupied | set(chosen)):
                chosen.append((int(r), int(c)))
                if len([p for p in chosen if p in set(outer_touch)]) >= MIN_MIDDLE_OUTER_TOUCH:
                    break
        if len([p for p in chosen if p in set(outer_touch)]) < MIN_MIDDLE_OUTER_TOUCH:
            continue

        pool = inner_touch[:]
        rng.shuffle(pool)
        for r, c in pool:
            if (int(r), int(c)) in chosen:
                continue
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in occupied | set(chosen)):
                chosen.append((int(r), int(c)))
                if len([p for p in chosen if p in set(inner_touch)]) >= MIN_MIDDLE_INNER_TOUCH:
                    break
        if len([p for p in chosen if p in set(inner_touch)]) < MIN_MIDDLE_INNER_TOUCH:
            continue

        remaining = [p for p in candidate_set if p not in chosen]
        rng.shuffle(remaining)
        for r, c in remaining:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in occupied | set(chosen)):
                chosen.append((int(r), int(c)))
                if len(chosen) >= n:
                    return chosen
    return None


def _expand_role_to_pairs(
    centers: list[tuple[int, int]],
    host: int,
    clean: np.ndarray,
    colors: tuple[int, int, int],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    groups: list[list[tuple[int, int]]] = []
    for r, c in centers:
        cells = [(int(r), int(c))]
        if rng.random() < DENT_PAIR_PROB:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                    continue
                if (int(nr), int(nc)) in occupied:
                    continue
                if int(clean[nr, nc]) != int(host):
                    continue
                if not _is_safe_dent_center(clean, int(nr), int(nc), int(host), colors):
                    continue
                if any(
                    max(abs(int(nr) - rr), abs(int(nc) - cc)) < DENT_PATCH_SEPARATION
                    for rr, cc in occupied
                    if (rr, cc) != (int(r), int(c))
                ):
                    continue
                cells.append((int(nr), int(nc)))
                occupied.add((int(nr), int(nc)))
                break
        groups.append(cells)
    return groups


def _stamp_dents(
    clean: np.ndarray,
    dent_groups: dict[int, list[list[tuple[int, int]]]],
    stamp_map: dict[int, int],
    *,
    clipped: bool,
) -> np.ndarray:
    h, w = clean.shape
    out = clean.copy()
    for host, groups in dent_groups.items():
        stamp_color = int(stamp_map[int(host)])
        for cells in groups:
            for r, c in cells:
                if clipped:
                    for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
                        for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                            if int(clean[rr, cc]) == int(host):
                                out[rr, cc] = int(stamp_color)
                else:
                    out[max(0, int(r) - 1) : min(h, int(r) + 2), max(0, int(c) - 1) : min(w, int(c) + 2)] = int(
                        stamp_color
                    )
    for groups in dent_groups.values():
        for cells in groups:
            for r, c in cells:
                out[int(r), int(c)] = int(Color.TEAL)
    return out


def _nearby_speck_color(clean: np.ndarray, r: int, c: int, colors: tuple[int, int, int], rng: random.Random) -> int:
    nearby = [int(x) for x in _sub3x3(clean, int(r), int(c)).flat if int(x) in colors]
    if nearby:
        return int(_Counter(nearby).most_common(1)[0][0])
    return int(rng.choice(list(colors)))


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, colors: tuple[int, int, int], rng: random.Random) -> int:
    bg_candidates = list(np.argwhere(clean == int(Color.BLACK)))
    rng.shuffle(bg_candidates)

    placed = 0
    for r, c in bg_candidates:
        rr = int(r)
        cc = int(c)
        if placed >= n_in_bg:
            break
        if int(inp[rr, cc]) != int(Color.BLACK):
            continue
        if int(np.sum(_sub3x3(inp, rr, cc) == int(Color.BLACK))) < BG_SPECK_MIN_BG_3x3:
            continue
        inp[rr, cc] = int(_nearby_speck_color(clean, rr, cc, colors, rng))
        placed += 1
    return placed


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[int(r) - rr, int(c) - cc] = int(color)
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

    sub_now = _sub3x3(inp, int(r), int(c))
    host_count = int(np.sum(sub_now == int(host)))
    other_count = int(np.sum(sub_now == int(other)))
    if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
        return False
    if _would_form_2x2(inp, int(r), int(c), int(other)):
        return False

    test = inp.copy()
    test[int(r), int(c)] = int(other)
    if _clean_cell(test, int(r), int(c), colors, bg) != int(host):
        return False

    for rr in range(max(0, int(r) - 1), min(inp.shape[0], int(r) + 2)):
        for cc in range(max(0, int(c) - 1), min(inp.shape[1], int(c) + 2)):
            before = _clean_cell(inp, int(rr), int(cc), colors, bg)
            after = _clean_cell(test, int(rr), int(cc), colors, bg)
            if (int(rr), int(cc)) == (int(r), int(c)):
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
            nearby = sorted({int(x) for x in _sub3x3(clean, int(r), int(c)).flat if int(x) in colors and int(x) != int(host)})
            for other in nearby:
                candidates.append((int(r), int(c), int(host), int(other)))
    rng.shuffle(candidates)

    placed = 0
    for r, c, host, other in candidates:
        if placed >= n_at_boundary:
            break
        if _boundary_noise_safe(inp, int(r), int(c), int(host), int(other), colors, int(Color.BLACK)):
            inp[int(r), int(c)] = int(other)
            placed += 1
    return placed


def _collect_candidates(
    clean: np.ndarray, role_colors: dict[str, int]
) -> tuple[tuple[int, int, int], dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]]]:
    colors = (int(role_colors["outer"]), int(role_colors["middle"]), int(role_colors["inner"]))
    valid = {"outer": [], "middle": [], "inner": []}
    categories = {
        "outer_clipped": [],
        "outer_interior": [],
        "middle_outer_touch": [],
        "middle_inner_touch": [],
        "middle_clipped": [],
        "inner_clipped": [],
        "inner_interior": [],
    }

    outer = int(role_colors["outer"])
    middle = int(role_colors["middle"])
    inner = int(role_colors["inner"])
    h, w = clean.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            host = int(clean[r, c])
            if host == outer and _is_safe_dent_center(clean, r, c, outer, colors):
                valid["outer"].append((int(r), int(c)))
                loss = _clip_loss(clean, r, c, outer)
                if loss > 0:
                    categories["outer_clipped"].append((int(r), int(c)))
                else:
                    categories["outer_interior"].append((int(r), int(c)))
            elif host == middle and _is_safe_dent_center(clean, r, c, middle, colors):
                valid["middle"].append((int(r), int(c)))
                loss = _clip_loss(clean, r, c, middle)
                if loss > 0:
                    categories["middle_clipped"].append((int(r), int(c)))
                if _middle_touches_outer(clean, r, c, outer):
                    categories["middle_outer_touch"].append((int(r), int(c)))
                if _middle_touches_inner(clean, r, c, inner):
                    categories["middle_inner_touch"].append((int(r), int(c)))
            elif host == inner and _is_safe_dent_center(clean, r, c, inner, colors):
                valid["inner"].append((int(r), int(c)))
                loss = _clip_loss(clean, r, c, inner)
                if loss > 0:
                    categories["inner_clipped"].append((int(r), int(c)))
                else:
                    categories["inner_interior"].append((int(r), int(c)))
    return colors, valid, categories


def _dent_metrics(
    clean: np.ndarray, dent_groups: dict[int, list[list[tuple[int, int]]]], role_colors: dict[str, int]
) -> dict[str, int]:
    outer = int(role_colors["outer"])
    middle = int(role_colors["middle"])
    inner = int(role_colors["inner"])
    metrics = {
        "outer_clipped": 0,
        "outer_interior": 0,
        "middle_clipped": 0,
        "middle_outer_touch": 0,
        "middle_inner_touch": 0,
        "inner_clipped": 0,
        "inner_total": 0,
        "total_clip_loss": 0,
    }
    for host, groups in dent_groups.items():
        for cells in groups:
            for r, c in cells:
                loss = _clip_loss(clean, int(r), int(c), int(host))
                metrics["total_clip_loss"] += int(loss)
                if int(host) == outer:
                    if loss > 0:
                        metrics["outer_clipped"] += 1
                    else:
                        metrics["outer_interior"] += 1
                elif int(host) == middle:
                    if loss > 0:
                        metrics["middle_clipped"] += 1
                    if _middle_touches_outer(clean, int(r), int(c), outer):
                        metrics["middle_outer_touch"] += 1
                    if _middle_touches_inner(clean, int(r), int(c), inner):
                        metrics["middle_inner_touch"] += 1
                elif int(host) == inner:
                    metrics["inner_total"] += 1
                    if loss > 0:
                        metrics["inner_clipped"] += 1
    return metrics


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [int(c) for c in Color.NOT_BLACK if int(c) != int(Color.TEAL)]
    outer_color, middle_color, inner_color = rng.sample(fg_pool, 3)
    role_colors = {"outer": int(outer_color), "middle": int(middle_color), "inner": int(inner_color)}

    clean = _build_regions(N, role_colors, rng)
    if clean is None:
        return None

    colors, valid, categories = _collect_candidates(clean, role_colors)
    wanted = {
        "outer": rng.randint(*OUTER_DENTS_RANGE),
        "middle": rng.randint(*MIDDLE_DENTS_RANGE),
        "inner": rng.randint(*INNER_DENTS_RANGE),
    }
    min_inner_clipped = max(3, int(np.ceil(wanted["inner"] * MIN_INNER_CLIPPED_FRAC)))

    if len(valid["outer"]) < wanted["outer"] or len(valid["middle"]) < wanted["middle"] or len(valid["inner"]) < wanted["inner"]:
        return None
    if len(categories["outer_clipped"]) < MIN_OUTER_CLIPPED or len(categories["outer_interior"]) < MIN_OUTER_INTERIOR:
        return None
    if (
        len(categories["middle_outer_touch"]) < MIN_MIDDLE_OUTER_TOUCH
        or len(categories["middle_inner_touch"]) < MIN_MIDDLE_INNER_TOUCH
        or len(categories["middle_clipped"]) < MIN_MIDDLE_CLIPPED
    ):
        return None
    if len(categories["inner_clipped"]) < min_inner_clipped:
        return None

    occupied: set[tuple[int, int]] = set()
    chosen: dict[str, list[tuple[int, int]]] = {}
    picked_middle = _pick_middle_centers(
        valid["middle"],
        wanted["middle"],
        rng,
        occupied,
        categories["middle_outer_touch"],
        categories["middle_inner_touch"],
    )
    if picked_middle is None:
        return None
    chosen["middle"] = picked_middle
    occupied.update(picked_middle)

    for role, preferred, min_pref in (
        ("inner", categories["inner_clipped"], min_inner_clipped),
        ("outer", categories["outer_clipped"], MIN_OUTER_CLIPPED),
    ):
        picked = _pick_centers(valid[role], wanted[role], rng, occupied, preferred=preferred, min_preferred=min_pref)
        if picked is None:
            return None
        chosen[role] = picked
        occupied.update(picked)

    if sum(1 for rc in chosen["outer"] if rc in set(categories["outer_interior"])) < MIN_OUTER_INTERIOR:
        return None

    occupied_for_pairs = set(occupied)
    dent_groups = {
        int(role_colors["outer"]): _expand_role_to_pairs(chosen["outer"], int(role_colors["outer"]), clean, colors, occupied_for_pairs, rng),
        int(role_colors["middle"]): _expand_role_to_pairs(chosen["middle"], int(role_colors["middle"]), clean, colors, occupied_for_pairs, rng),
        int(role_colors["inner"]): _expand_role_to_pairs(chosen["inner"], int(role_colors["inner"]), clean, colors, occupied_for_pairs, rng),
    }

    metrics = _dent_metrics(clean, dent_groups, role_colors)
    if metrics["outer_clipped"] < MIN_OUTER_CLIPPED or metrics["outer_interior"] < MIN_OUTER_INTERIOR:
        return None
    if (
        metrics["middle_outer_touch"] < MIN_MIDDLE_OUTER_TOUCH
        or metrics["middle_inner_touch"] < MIN_MIDDLE_INNER_TOUCH
        or metrics["middle_clipped"] < MIN_MIDDLE_CLIPPED
    ):
        return None
    if metrics["inner_total"] <= 0 or metrics["inner_clipped"] < max(3, int(np.ceil(metrics["inner_total"] * MIN_INNER_CLIPPED_FRAC))):
        return None
    if metrics["total_clip_loss"] < MIN_TOTAL_CLIP_LOSS:
        return None

    stamp_map = _stamp_map(role_colors)
    output = _stamp_dents(clean, dent_groups, stamp_map, clipped=True)
    legacy = _stamp_dents(clean, dent_groups, stamp_map, clipped=False)
    if np.array_equal(output, legacy):
        return None

    inp = clean.copy()
    for groups in dent_groups.values():
        for cells in groups:
            for r, c in cells:
                inp[int(r), int(c)] = int(Color.BLACK)

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    placed_bg = _add_bg_noise(inp, clean, n_bg, colors, rng)
    placed_bd = _add_boundary_noise(inp, clean, n_noise - placed_bg, colors, rng)
    if placed_bg + placed_bd < MIN_NOISE_PLACED:
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
