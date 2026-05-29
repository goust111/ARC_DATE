# Combo variant of de809cff (index 35)
# combo_spec: D8/0+D13/2
# Name: 单向映射+背景反向凹痕
#
# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The clean scene still has two large foreground regions on black background.
#    - Foreground dents are still black holes punched into those regions, but only one
#      foreground host is active: the larger cleaned foreground region by visible area.
#    - Background dents are the reverse pattern: a lone foreground-colored cell sits in a
#      background-majority 3x3 neighborhood, gets cleaned back to black, then that black
#      host stamps a 3x3 black patch with a teal center.
#    - Small trace:
#        base task: black hole in region A -> opposite-color 3x3 patch, black hole in
#        region B -> opposite-color 3x3 patch, fg speck in background -> disappears.
#        combo: black hole in larger region -> opposite-color patch with teal center;
#        black hole in smaller region -> remains a black hole; fg speck in background
#        -> becomes a teal-centered black 3x3 bite if the cleaned 3x3 is bg-dominant.
#      The two edits interact because the solver must infer both "which fg host is active"
#      and "background also counts as a host".
# b) Complexity calibration
#    - The generator forces repeated evidence of all three cases in each sample:
#      active fg dents, inactive fg dents, and background reverse dents.
#    - The active/inactive choice is binary and repeated many times, while background dents
#      are kept to a smaller 2-5 count so the new signal is visible but not overwhelming.
#    - Disposable fg specks in background are removed from the noise model; otherwise the
#      solver would have to separate meaningful and meaningless fg-in-bg cells from too
#      little evidence.
# c) Implicit assumption review
#    - The base `_dent_centers(...)` only looked at original black cells, which fails once
#      background dents start as foreground-colored cells.
#    - The base `main(...)` stamped every foreground dent symmetrically. That assumption
#      breaks under the one-way mapping; inactive-host dents must be restored to black.
#    - Starting from `clean.copy()` heals every dent, so the combo solver must explicitly
#      re-black inactive dents after stamping active and background events.
# d) Placement legality
#    - All dent centers, including background dents, obey Chebyshev separation >= 3 so their
#      3x3 effects do not overlap.
#    - Background dents are chosen only from interior black cells whose clean 3x3 is uniquely
#      bg-dominant with 1-3 nearby foreground cells; this makes the black patch visible.
#    - Boundary-noise cells are kept away from dent cells so cleanup cannot perturb the dent
#      neighborhoods used by the solver.
# e) Visual observability
#    - The larger foreground region is forced to exceed the smaller by a clear visible margin,
#      so humans can infer the active host from area, not from color ids.
#    - Each sample visibly contains teal-marked black bites in background plus multiple black
#      holes that stay untouched inside the smaller fg region, so the combo behavior is not a
#      subtle edge case.
# f) Visual cleanliness
#    - Background dents remain single-cell centers only; paired dents are reserved for
#      foreground regions to keep the new background signal easy to parse.
#    - Because 3x3 patches never overlap, teal markers and black holes stay attributable to
#      one cause each.
# g) Color-change inferability
#    - Only two foreground colors plus black/teal appear.
#    - The color rule is structured by host type: larger fg host maps to the other fg color,
#      smaller fg host maps to nothing, and bg host maps to black. The mapping is therefore
#      inferable from region membership and area, not arbitrary color lookup.
# 9) New solver assumptions audited against the generator
#    - The solver assumes the active host is the larger cleaned foreground region; the
#      generator enforces a pre-noise area gap and re-checks the gap after cleanup.
#    - The solver assumes every real dent cell is black after cleanup; the generator only
#      inserts background reverse dents whose inserted color is guaranteed to clean back to
#      black.
#    - The solver assumes host-majority in the cleaned 3x3 is unique and at least 5; the
#      generator filters all foreground and background dent candidates with that condition.

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


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


def _clean_regions(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell corruption before detecting dents in any host region.

    The pass still only reasons about the two foreground colors. Background dents are
    recognized later from cells that become black after this cleanup.
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


def _dominant_host(sub: np.ndarray, colors: tuple[int, int], bg: int) -> tuple[int | None, dict[int, int]]:
    labels = (int(bg), int(colors[0]), int(colors[1]))
    counts = {label: int(np.sum(sub == label)) for label in labels}
    best_count = max(counts.values())
    if best_count < 5:
        return None, counts
    winners = [label for label in labels if counts[label] == best_count]
    if len(winners) != 1:
        return None, counts
    return int(winners[0]), counts


def _active_host_color(clean: np.ndarray, colors: tuple[int, int]) -> int:
    counts = {int(color): int(np.sum(clean == int(color))) for color in colors}
    return int(max(colors, key=lambda color: counts[int(color)]))


def _dent_events(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    """
    Return (r, c, host) for every true dent after cleanup.

    - host in foreground: original cell must have been black
    - host == background: original cell must have been one of the foreground colors
    """
    h, w = original.shape
    events: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(clean[r, c]) != int(bg):
                continue
            host, _counts = _dominant_host(_sub3x3(clean, r, c), colors, bg)
            if host is None:
                continue

            original_value = int(original[r, c])
            if int(host) == int(bg):
                if original_value in colors:
                    events.append((r, c, int(bg)))
            elif original_value == int(bg):
                events.append((r, c, int(host)))
    return events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply one-way foreground dents plus reverse dents in the background.

    Input structure:
    - Black background with two large foreground colors whose cleaned visible areas
      are intentionally unequal.
    - Both foreground regions may contain black dents, and the background may also
      contain isolated foreground-colored reverse dents near region edges.
    - A few one-cell cross-color contaminants can appear along the fg/fg boundary.

    Rule:
    - First clean the two foreground colors locally: one-cell boundary contamination
      reverts to its host, and isolated foreground specks revert to black.
    - Determine which foreground color owns the larger cleaned area. Only dents whose
      cleaned 3x3 neighborhood is uniquely dominated by that larger foreground host
      produce a 3x3 patch in the other foreground color, with the dent center teal.
    - A cleaned-black cell whose 3x3 neighborhood is uniquely dominated by black and
      whose original value was foreground-colored is a background reverse dent: stamp
      a 3x3 black patch there and mark its center teal. Dents in the smaller
      foreground region stay as black holes.

    Key insight:
    - The output is driven by host region after cleanup, but foreground hosts are now
      asymmetric. The larger foreground region is active, the smaller one is inert,
      and the background itself can also be a host that claims a 3x3 black patch.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_regions(g, colors, bg)
    active = _active_host_color(clean, colors)
    inactive = int(colors[1] if active == int(colors[0]) else colors[0])

    out = clean.copy()
    active_fg: list[tuple[int, int]] = []
    inactive_fg: list[tuple[int, int]] = []
    bg_events: list[tuple[int, int]] = []
    for r, c, host in _dent_events(g, clean, colors, bg):
        if int(host) == int(bg):
            bg_events.append((r, c))
        elif int(host) == int(active):
            active_fg.append((r, c))
        else:
            inactive_fg.append((r, c))

    h, w = g.shape
    for r, c in active_fg:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(inactive)
    for r, c in bg_events:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(bg)

    for r, c in active_fg:
        out[r, c] = Color.TEAL
    for r, c in bg_events:
        out[r, c] = Color.TEAL
    for r, c in inactive_fg:
        out[r, c] = int(bg)

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85
ACTIVE_AREA_GAP_MIN = 35
SOLVER_ACTIVE_AREA_GAP_MIN = 20

# --- CFG: dent placement ---
FG_DENT_COUNT_MIN, FG_DENT_COUNT_MAX = 10, 18
BG_DENT_COUNT_MIN, BG_DENT_COUNT_MAX = 2, 5
MIN_ACTIVE_DENTS = 5
MIN_INACTIVE_DENTS = 4
FG_DENT_MIN_HOST_3x3 = 5
BG_DENT_MIN_BG_3x3 = 6
BG_DENT_MAX_FG_3x3 = 3
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 4, 12
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300
CENTER_PICK_ATTEMPTS = 12


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles using a simple painter's algorithm."""
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


def _is_safe_fg_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """After setting (r,c) to black, the solver must detect a foreground-host dent."""
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    return host_count >= FG_DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == other))


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


def _far_enough(r: int, c: int, others: list[tuple[int, int]]) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in others)


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    blocked: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Select n centers with global Chebyshev separation."""
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    blocked = [] if blocked is None else list(blocked)
    for r, c in pool:
        if _far_enough(r, c, chosen) and _far_enough(r, c, blocked):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    """Optionally expand some foreground dents into 2-cell adjacent pairs."""
    h, w = clean.shape
    result: list[list[tuple[int, int]]] = []
    occupied = list(blocked)
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
                if not _is_safe_fg_dent_center(clean, nr, nc, host, other):
                    continue
                if all(
                    max(abs(nr - r2), abs(nc - c2)) >= DENT_PATCH_SEPARATION
                    for r2, c2 in occupied
                    if (r2, c2) != (r, c)
                ):
                    cells.append((nr, nc))
                    occupied.append((nr, nc))
                    break
        occupied.append((r, c))
        result.append(cells)
    return result


def _center_cleanup_result(
    clean: np.ndarray, r: int, c: int, value: int, colors: tuple[int, int], bg: int
) -> int:
    """Simulate the first cleanup sweep for the center cell after inserting `value`."""
    tmp = clean.copy()
    tmp[r, c] = int(value)
    sub = _sub3x3(tmp, r, c)
    own = int(np.sum(sub == int(value)))
    other = int(colors[1]) if int(value) == int(colors[0]) else int(colors[0])
    other_count = int(np.sum(sub == other))
    bg_count = int(np.sum(sub == int(bg)))

    if _has_same_2x2(tmp, r, c, int(value)):
        return int(value)
    if other_count > own and other_count >= 5:
        return int(other)
    if bg_count >= 5 and own <= 4:
        return int(bg)
    if own <= 2:
        return int(bg)
    return int(value)


def _bg_dent_color_options(
    clean: np.ndarray, r: int, c: int, colors: tuple[int, int], bg: int
) -> list[int]:
    """Return foreground colors that act as safe reverse dents in background."""
    h, w = clean.shape
    if not (0 < r < h - 1 and 0 < c < w - 1):
        return []
    if int(clean[r, c]) != int(bg):
        return []
    host, counts = _dominant_host(_sub3x3(clean, r, c), colors, bg)
    if host != int(bg):
        return []
    fg_total = counts[int(colors[0])] + counts[int(colors[1])]
    if not (1 <= fg_total <= BG_DENT_MAX_FG_3x3):
        return []
    if counts[int(bg)] < BG_DENT_MIN_BG_3x3:
        return []

    options = []
    for color in colors:
        if _center_cleanup_result(clean, r, c, int(color), colors, bg) == int(bg):
            options.append(int(color))
    return options


def _choose_bg_dent_color(
    clean: np.ndarray, r: int, c: int, colors: tuple[int, int], rng: random.Random
) -> int:
    options = _bg_dent_color_options(clean, r, c, colors, Color.BLACK)
    assert options, "background dent must have at least one safe inserted color"
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    best = max(options, key=lambda color: counts[int(color)])
    best_options = [int(color) for color in options if counts[int(color)] == counts[int(best)]]
    return int(rng.choice(best_options))


def _build_output(
    clean: np.ndarray,
    active_groups: list[list[tuple[int, int]]],
    inactive_groups: list[list[tuple[int, int]]],
    bg_centers: list[tuple[int, int]],
    inactive_color: int,
) -> np.ndarray:
    """Build expected output for the combo rule."""
    h, w = clean.shape
    output = clean.copy()
    for cells in active_groups:
        for r, c in cells:
            output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(inactive_color)
    for r, c in bg_centers:
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = Color.BLACK

    for cells in active_groups:
        for r, c in cells:
            output[r, c] = Color.TEAL
    for r, c in bg_centers:
        output[r, c] = Color.TEAL
    for cells in inactive_groups:
        for r, c in cells:
            output[r, c] = Color.BLACK
    return output


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """Check that changing (r,c) to `other` will not create cleanup cascades nearby."""
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


def _too_close_to_reserved(r: int, c: int, reserved: list[tuple[int, int]], radius: int = 1) -> bool:
    return any(max(abs(r - rr), abs(c - cc)) <= radius for rr, cc in reserved)


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    reserved: list[tuple[int, int]],
    rng: random.Random,
) -> None:
    """Place fg-vs-fg contamination near the shared boundary, away from dents."""
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if v == int(color_a) else int(color_a)
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1 and not _too_close_to_reserved(r, c, reserved):
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


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    counts = {int(color_a): int(np.sum(clean == color_a)), int(color_b): int(np.sum(clean == color_b))}
    if abs(counts[int(color_a)] - counts[int(color_b)]) < ACTIVE_AREA_GAP_MIN:
        return None
    active_color = int(max((int(color_a), int(color_b)), key=lambda color: counts[int(color)]))
    inactive_color = int(color_b if active_color == int(color_a) else color_a)

    n_fg = rng.randint(FG_DENT_COUNT_MIN, FG_DENT_COUNT_MAX)
    n_active_min = max(MIN_ACTIVE_DENTS, n_fg // 2)
    n_active_max = n_fg - MIN_INACTIVE_DENTS
    if n_active_min > n_active_max:
        return None
    n_active = rng.randint(n_active_min, n_active_max)
    n_inactive = n_fg - n_active
    n_bg = rng.randint(BG_DENT_COUNT_MIN, BG_DENT_COUNT_MAX)

    h, w = clean.shape
    valid_fg = {int(color_a): [], int(color_b): []}
    bg_options: dict[tuple[int, int], list[int]] = {}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == int(color_a) and _is_safe_fg_dent_center(clean, r, c, color_a, color_b):
                valid_fg[int(color_a)].append((r, c))
            elif v == int(color_b) and _is_safe_fg_dent_center(clean, r, c, color_b, color_a):
                valid_fg[int(color_b)].append((r, c))
            elif v == Color.BLACK:
                options = _bg_dent_color_options(clean, r, c, colors, Color.BLACK)
                if options:
                    bg_options[(r, c)] = options

    if len(valid_fg[active_color]) < n_active + 2 or len(valid_fg[inactive_color]) < n_inactive + 2:
        return None
    if len(bg_options) < n_bg:
        return None

    active_centers = None
    inactive_centers = None
    bg_centers = None
    for _ in range(CENTER_PICK_ATTEMPTS):
        active_centers = _pick_dent_centers(valid_fg[active_color], n_active, rng)
        if active_centers is None:
            continue
        inactive_centers = _pick_dent_centers(valid_fg[inactive_color], n_inactive, rng, blocked=active_centers)
        if inactive_centers is None:
            continue
        bg_centers = _pick_dent_centers(list(bg_options), n_bg, rng, blocked=active_centers + inactive_centers)
        if bg_centers is not None:
            break
    if active_centers is None or inactive_centers is None or bg_centers is None:
        return None

    base_centers = active_centers + inactive_centers + bg_centers
    active_groups = _expand_to_pairs(active_centers, active_color, inactive_color, clean, base_centers, rng)
    extra_active = [cell for cells in active_groups for cell in cells[1:]]
    inactive_groups = _expand_to_pairs(
        inactive_centers, inactive_color, active_color, clean, base_centers + extra_active, rng
    )

    output = _build_output(clean, active_groups, inactive_groups, bg_centers, inactive_color)

    inp = clean.copy()
    for cells in active_groups + inactive_groups:
        for r, c in cells:
            inp[r, c] = Color.BLACK
    for r, c in bg_centers:
        inp[r, c] = _choose_bg_dent_color(clean, r, c, colors, rng)

    dent_cells = [cell for cells in active_groups + inactive_groups for cell in cells] + bg_centers
    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    _add_boundary_noise(inp, clean, n_noise, color_a, color_b, dent_cells, rng)

    solver_clean = _clean_regions(inp, colors, Color.BLACK)
    resolved_active = _active_host_color(solver_clean, colors)
    resolved_inactive = int(color_b if resolved_active == int(color_a) else color_a)
    if resolved_active != int(active_color):
        return None
    if abs(int(np.sum(solver_clean == resolved_active)) - int(np.sum(solver_clean == resolved_inactive))) < SOLVER_ACTIVE_AREA_GAP_MIN:
        return None

    events = _dent_events(inp, solver_clean, colors, Color.BLACK)
    active_event_count = sum(1 for _r, _c, host in events if int(host) == int(resolved_active))
    inactive_event_count = sum(1 for _r, _c, host in events if int(host) == int(resolved_inactive))
    bg_event_count = sum(1 for _r, _c, host in events if int(host) == Color.BLACK)
    if active_event_count < MIN_ACTIVE_DENTS or inactive_event_count < MIN_INACTIVE_DENTS:
        return None
    if bg_event_count < n_bg:
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
