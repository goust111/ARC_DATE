# Combo variant of de809cff (index 8)
# combo_spec: D8/0+D17/1
# Name: 背景区域含反向凹痕+凹痕恰在区域边界上

from __future__ import annotations

# Design analysis for the D8/0 + D17/1 combo
# -------------------------------------------
# a) Concrete behavior and trace:
#    The base task treats only black cells inside one foreground region as dents.
#    This combo adds a third host type (background) and simultaneously forces every
#    foreground dent to sit on the frontier between the two foreground colors.
#    Small local trace, with `a`/`b` the two foreground colors and `x` a reverse dent:
#      a a b          a b 0
#      a . b    ->    b 8 b
#      a a b          b b b
#    The black `.` is not an interior hole anymore: it belongs to color `a`, but its
#    3x3 already contains both foreground colors and an orthogonal `b` neighbor, so the
#    solver must detect "boundary dent with `a` as local majority" and stamp color `b`.
#    In the same grid a background reverse dent can look like:
#      0 0 a          0 0 0
#      0 x 0    ->    0 8 0
#      0 0 0          0 0 0
#    where the inserted foreground pixel `x` is cleaned back to black and then expands
#    as a black 3x3 patch because the host region is the background.
# b) Complexity calibration:
#    To keep the combo readable, there are still only two foreground colors plus black
#    and teal. The new difficulty is structural, not palette-based: every sample must
#    repeatedly show (1) black dents exactly on the fg-fg boundary and (2) fg-colored
#    single pixels inside the background that also turn into teal-centered patches.
#    Ordinary background speck noise is removed from the generator so that every fg-in-bg
#    singleton supports the new D8 rule instead of introducing ambiguous exceptions.
# c) Hidden assumptions reviewed:
#    - The base `_dent_centers` iterates only over original black cells. That breaks as
#      soon as background reverse dents exist, because their original center is colored.
#    - The base foreground logic accepts interior dents. D17/1 invalidates that: a true
#      foreground event must now have the other foreground color in its cleaned 3x3 and
#      must touch it orthogonally, otherwise it is not a boundary dent.
#    - The base writeback "stamp the opposite foreground color" still holds for
#      foreground-host dents, but background-host dents must stamp black instead.
# d) Placement legality:
#    The generator keeps a global Chebyshev separation of at least 3 between primary
#    dent centers so 3x3 effects stay disambiguated. Boundary noise is placed away from
#    reserved dent cells, and reverse dents are restricted to background cells whose
#    local cleanup deterministically returns them to black.
# e) Visual observability:
#    Solvers can infer the combo from 2-4 examples because all three host types appear
#    repeatedly: color-A boundary dents, color-B boundary dents, and background reverse
#    dents. The boundary dents are visible because their 3x3 neighborhoods always mix
#    both foreground colors, unlike the original task's frequent interior holes.
# f) Visual cleanliness:
#    The generator avoids overlapping 3x3 patches and avoids disposable fg specks in the
#    background. That keeps teal centers legible and prevents the background-dent signal
#    from collapsing into arbitrary cleanup noise.
# g) Color-change inferability:
#    The rule remains host-driven and consistent: foreground hosts stamp the opposite
#    foreground color; background hosts stamp black; every center becomes teal. No logic
#    depends on numeric color ids, only on visible local majority and adjacency.
# 9) New solver assumptions audited against the generator:
#    - Every detected event cell becomes black after cleanup; the generator guarantees
#      that for both black dents and reverse dents.
#    - Every foreground event is truly on the fg-fg interface; the generator enforces an
#      orthogonal neighbor of the opposite foreground color and a mixed 3x3.
#    - Every background reverse dent has a unique black-majority 3x3 after cleanup and
#      at least one nearby foreground cell, so the black patch is visually meaningful.

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


def _touches_orthogonal(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(color):
            return True
    return False


def _clean_regions(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting dents in any host region.

    Two sweeps are enough: the first fixes the one-cell corruption, the second
    stabilizes neighboring cells around that corrected local majority.
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


def _dent_events(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int, int]]:
    """
    Return (r, c, host, stamp_color) for all active dents.

    Foreground dents are valid only when the cleaned 3x3 is a mixed foreground
    boundary neighborhood and the opposite foreground color touches orthogonally.
    Background reverse dents are colored singletons that clean back to black and
    sit in a uniquely black-majority 3x3.
    """
    h, w = original.shape
    events: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(clean[r, c]) != int(bg):
                continue
            host, counts = _dominant_host(_sub3x3(clean, r, c), colors, bg)
            if host is None:
                continue

            v = int(original[r, c])
            if int(host) == int(bg):
                if v in colors:
                    events.append((r, c, int(bg), int(bg)))
                continue

            if v != int(bg):
                continue
            other = int(colors[1]) if int(host) == int(colors[0]) else int(colors[0])
            if counts[int(other)] < 1:
                continue
            if not _touches_orthogonal(clean, r, c, int(other)):
                continue
            events.append((r, c, int(host), int(other)))
    return events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn boundary dents and background reverse dents into teal-centered 3x3 patches.

    Input structure:
    - Black background with two large foreground colors.
    - The foreground defects are black holes located exactly on the boundary between
      the two foreground regions, so their cleaned 3x3 neighborhoods contain both colors.
    - The background can also contain isolated foreground-colored reverse dents, and
      there may be a few one-cell foreground contaminations near the fg-fg boundary.

    Rule:
    - First clean the two foreground colors locally: isolated foreground specks that
      do not belong to a stable region disappear into black, and one-cell cross-color
      contamination is reassigned to the surrounding foreground majority.
    - Then inspect every cell that becomes black after cleanup. If its cleaned 3x3
      has a unique black majority and the original cell was foreground-colored, it is
      a background reverse dent and stamps a black 3x3 patch.
    - Otherwise, if its cleaned 3x3 has a unique foreground majority, the original
      cell was black, and the opposite foreground color touches that cell orthogonally,
      it is a true boundary dent of that majority region and stamps the other foreground
      color on its 3x3 neighborhood. Every detected center becomes teal (8).

    Key insight:
    - Cleanup reveals which of the three region types really owned the defect cell.
      A black-majority neighborhood means the background was interrupted by a stray
      foreground pixel; a mixed foreground neighborhood with one dominant color means
      a hole was punched out of that boundary pixel of the dominant foreground region.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_regions(g, colors, bg)

    out = clean.copy()
    events = _dent_events(g, clean, colors, bg)
    h, w = g.shape
    for r, c, _host, stamp_color in events:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)

    for r, c, _host, _stamp_color in events:
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
FG_DENT_COUNT_MIN, FG_DENT_COUNT_MAX = 4, 8
BG_DENT_COUNT_MIN, BG_DENT_COUNT_MAX = 2, 5
FG_DENT_MIN_HOST_3x3 = 5
FG_DENT_MIN_OTHER_3x3 = 1
FG_DENT_MAX_BG_3x3 = 1
BG_DENT_MIN_BG_3x3 = 6
BG_DENT_MAX_FG_3x3 = 3
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.05
MIN_STRONG_BOUNDARY_DENTS_PER_COLOR = 1

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 1, 4
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400
SELECT_RETRY_ATTEMPTS = 120


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint two diagonal rectangles so the fg-fg interface is long and stable."""

    def _make_rect(corner: str) -> tuple[int, int, int, int]:
        h = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        w = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        m = rng.randint(0, RECT_MARGIN_MAX)
        r0 = m if corner[0] == "T" else max(0, N - h - m)
        c0 = m if corner[1] == "L" else max(0, N - w - m)
        return r0, c0, min(N, r0 + h), min(N, c0 + w)

    for _ in range(18):
        corner_pair = rng.choice([("TL", "BR"), ("TR", "BL")])
        rect_specs = [
            (*_make_rect(corner_pair[0]), int(color_a)),
            (*_make_rect(corner_pair[1]), int(color_b)),
        ]
        if rng.random() < 0.5:
            rect_specs.reverse()

        grid = np.zeros((N, N), dtype=int)
        for r0, c0, r1, c1, color in rect_specs:
            grid[r0:r1, c0:c1] = int(color)

        cnt_a = int(np.sum(grid == int(color_a)))
        cnt_b = int(np.sum(grid == int(color_b)))
        if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
            continue
        coverage = (cnt_a + cnt_b) / float(N * N)
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            continue
        r0a, c0a, r1a, c1a, _ = rect_specs[0]
        r0b, c0b, r1b, c1b, _ = rect_specs[1]
        overlap_h = max(0, min(r1a, r1b) - max(r0a, r0b))
        overlap_w = max(0, min(c1a, c1b) - max(c0a, c0b))
        if overlap_h * overlap_w < 16:
            continue
        interface = 0
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                v = int(grid[r, c])
                if v not in (int(color_a), int(color_b)):
                    continue
                other = int(color_b) if v == int(color_a) else int(color_a)
                if _touches_orthogonal(grid, r, c, other):
                    interface += 1
        if interface >= 16:
            return grid
    return None


def _is_safe_boundary_fg_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """A valid foreground dent must be a mixed fg-fg boundary cell with unique host majority."""
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    other_count = int(np.sum(sub == int(other)))
    bg_count = int(np.sum(sub == Color.BLACK))
    if host_count < FG_DENT_MIN_HOST_3x3 or host_count <= other_count:
        return False
    if other_count < FG_DENT_MIN_OTHER_3x3 or bg_count > FG_DENT_MAX_BG_3x3:
        return False
    return _touches_orthogonal(grid, r, c, int(other))


def _is_strong_boundary_fg_dent(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    sub = _sub3x3(grid, r, c)
    return _is_safe_boundary_fg_dent_center(grid, r, c, host, other) and int(np.sum(sub == int(other))) >= 2


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = int(color)
                if int(np.sum(block == int(color))) == 4:
                    return True
    return False


def _center_cleanup_result(
    clean: np.ndarray, r: int, c: int, value: int, colors: tuple[int, int], bg: int
) -> int:
    """Simulate one cleanup decision for the inserted center cell."""
    tmp = clean.copy()
    tmp[r, c] = int(value)
    sub = _sub3x3(tmp, r, c)
    own = int(np.sum(sub == int(value)))
    other = int(colors[1]) if int(value) == int(colors[0]) else int(colors[0])
    other_count = int(np.sum(sub == int(other)))
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
    """Return foreground colors that behave as safe reverse dents in background."""
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

    options: list[int] = []
    for color in colors:
        if _center_cleanup_result(clean, r, c, int(color), colors, bg) == int(bg):
            options.append(int(color))
    return options


def _choose_bg_dent_color(
    clean: np.ndarray, r: int, c: int, colors: tuple[int, int], rng: random.Random
) -> int:
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    options = _bg_dent_color_options(clean, r, c, colors, Color.BLACK)
    assert options, "background dent must have at least one safe inserted color"
    best = max(options, key=lambda color: counts[int(color)])
    tied = [int(color) for color in options if counts[int(color)] == counts[int(best)]]
    return int(rng.choice(tied))


def _spaced_enough(candidate: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = candidate
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen)


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    *,
    blocked: list[tuple[int, int]] | None = None,
    preferred_groups: list[tuple[list[tuple[int, int]], int]] | None = None,
) -> list[tuple[int, int]] | None:
    blocked = list(blocked or [])
    preferred_groups = list(preferred_groups or [])
    candidate_set = set(candidates)
    filtered_groups = [([p for p in group if p in candidate_set], int(target)) for group, target in preferred_groups]
    if sum(target for _group, target in filtered_groups) > n:
        return None

    for _ in range(SELECT_RETRY_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        ok = True

        for group, target in filtered_groups:
            order = list(group)
            rng.shuffle(order)
            picked = 0
            for candidate in order:
                if candidate in used:
                    continue
                if _spaced_enough(candidate, blocked + chosen):
                    chosen.append(candidate)
                    used.add(candidate)
                    picked += 1
                    if picked >= target:
                        break
            if picked < target:
                ok = False
                break
        if not ok:
            continue

        rest = [candidate for candidate in candidates if candidate not in used]
        rng.shuffle(rest)
        for candidate in rest:
            if _spaced_enough(candidate, blocked + chosen):
                chosen.append(candidate)
                if len(chosen) >= n:
                    return chosen
    return None


def _expand_to_boundary_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    """Optionally expand some boundary dents into 2-cell orthogonal pairs."""
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
                if not _is_safe_boundary_fg_dent_center(clean, nr, nc, host, other):
                    continue
                if all(
                    max(abs(nr - rr), abs(nc - cc)) >= DENT_PATCH_SEPARATION
                    for rr, cc in occupied
                    if (rr, cc) != (r, c)
                ):
                    cells.append((nr, nc))
                    occupied.append((nr, nc))
                    break
        occupied.append((r, c))
        result.append(cells)
    return result


def _stamp_dents(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    dent_centers_bg: list[tuple[int, int]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    """Build output by stamping host-dependent 3x3 patches and marking teal centers."""
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for r, c in dent_centers_bg:
        output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = Color.BLACK

    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    for r, c in dent_centers_bg:
        output[r, c] = Color.TEAL
    return output


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """Check that changing (r,c) to `other` will not trigger cleanup cascades nearby."""
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
            bg_n = int(np.sum(nsub == Color.BLACK))
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
    """Place one-cell cross-color contamination along the fg-fg interface."""
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if v == int(color_a) else int(color_a)
            if _touches_orthogonal(clean, r, c, other) and not _too_close_to_reserved(r, c, reserved, radius=2):
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
        if _would_form_2x2(inp, r, c, int(other)):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, int(host), int(other)):
            continue
        inp[r, c] = int(other)
        placed += 1


def _flatten_groups(groups: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for cells in groups for cell in cells]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_fg = rng.randint(FG_DENT_COUNT_MIN, FG_DENT_COUNT_MAX)
    n_bg = rng.randint(BG_DENT_COUNT_MIN, BG_DENT_COUNT_MAX)
    n_a, n_b = n_fg // 2, n_fg - n_fg // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    h, w = clean.shape
    valid_fg: dict[int, list[tuple[int, int]]] = {int(color_a): [], int(color_b): []}
    strong_fg: dict[int, list[tuple[int, int]]] = {int(color_a): [], int(color_b): []}
    bg_options: dict[tuple[int, int], list[int]] = {}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == int(color_a):
                if _is_safe_boundary_fg_dent_center(clean, r, c, int(color_a), int(color_b)):
                    valid_fg[int(color_a)].append((r, c))
                    if _is_strong_boundary_fg_dent(clean, r, c, int(color_a), int(color_b)):
                        strong_fg[int(color_a)].append((r, c))
            elif v == int(color_b):
                if _is_safe_boundary_fg_dent_center(clean, r, c, int(color_b), int(color_a)):
                    valid_fg[int(color_b)].append((r, c))
                    if _is_strong_boundary_fg_dent(clean, r, c, int(color_b), int(color_a)):
                        strong_fg[int(color_b)].append((r, c))
            elif v == Color.BLACK:
                options = _bg_dent_color_options(clean, r, c, colors, Color.BLACK)
                if options:
                    bg_options[(r, c)] = options

    if len(valid_fg[int(color_a)]) < n_a + 1 or len(valid_fg[int(color_b)]) < n_b + 1:
        return None
    if len(strong_fg[int(color_a)]) < min(MIN_STRONG_BOUNDARY_DENTS_PER_COLOR, n_a):
        return None
    if len(strong_fg[int(color_b)]) < min(MIN_STRONG_BOUNDARY_DENTS_PER_COLOR, n_b):
        return None
    if len(bg_options) < n_bg:
        return None

    specs = [
        {
            "color": int(color_a),
            "other": int(color_b),
            "need": n_a,
            "valid": valid_fg[int(color_a)],
            "strong": strong_fg[int(color_a)],
        },
        {
            "color": int(color_b),
            "other": int(color_a),
            "need": n_b,
            "valid": valid_fg[int(color_b)],
            "strong": strong_fg[int(color_b)],
        },
    ]
    specs.sort(key=lambda spec: (len(spec["valid"]) - int(spec["need"]), len(spec["valid"])))

    chosen_fg: dict[int, list[tuple[int, int]]] = {}
    blocked_fg: list[tuple[int, int]] = []
    for spec in specs:
        chosen = _pick_dent_centers(
            spec["valid"],
            int(spec["need"]),
            rng,
            blocked=blocked_fg,
            preferred_groups=[(spec["strong"], min(MIN_STRONG_BOUNDARY_DENTS_PER_COLOR, int(spec["need"])))],
        )
        if chosen is None:
            return None
        chosen_fg[int(spec["color"])] = chosen
        blocked_fg += chosen

    dents_a = chosen_fg[int(color_a)]
    dents_b = chosen_fg[int(color_b)]
    dents_bg = _pick_dent_centers(list(bg_options), n_bg, rng, blocked=dents_a + dents_b)
    if dents_bg is None:
        return None

    base_centers = dents_a + dents_b + dents_bg
    groups_a = _expand_to_boundary_pairs(dents_a, int(color_a), int(color_b), clean, base_centers, rng)
    extra_a = [cell for cells in groups_a for cell in cells[1:]]
    groups_b = _expand_to_boundary_pairs(
        dents_b, int(color_b), int(color_a), clean, base_centers + extra_a, rng
    )

    output = _stamp_dents(clean, groups_a, groups_b, dents_bg, int(color_a), int(color_b))

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = Color.BLACK
    for r, c in dents_bg:
        inp[r, c] = _choose_bg_dent_color(clean, r, c, colors, rng)

    dent_cells = _flatten_groups(groups_a) + _flatten_groups(groups_b) + dents_bg
    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    _add_boundary_noise(inp, clean, n_noise, int(color_a), int(color_b), dent_cells, rng)

    try:
        clean_after = _clean_regions(inp, colors, Color.BLACK)
        events = _dent_events(inp, clean_after, colors, Color.BLACK)
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None

    boundary_events = 0
    bg_events = 0
    for r, c, host, _stamp in events:
        if host == Color.BLACK:
            bg_events += 1
        elif host == int(color_a):
            if not _touches_orthogonal(clean_after, r, c, int(color_b)):
                return None
            boundary_events += 1
        elif host == int(color_b):
            if not _touches_orthogonal(clean_after, r, c, int(color_a)):
                return None
            boundary_events += 1

    if boundary_events < len(_flatten_groups(groups_a)) + len(_flatten_groups(groups_b)):
        return None
    if bg_events < n_bg:
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
