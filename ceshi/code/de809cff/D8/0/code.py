# Variant of de809cff
# Dimension: D8 — 背景区域的角色 (meta)
# Change: 背景色（0）区域不参与变换，仅作为前景区域之间的分隔空间和网格边缘填充... → 背景区域也包含凹痕（前景色凹痕）
#
# Design analysis for the D8 variant
# a) Concrete behavior and trace
#    - The clean scene still contains two large foreground regions on black background.
#    - Foreground dents remain as before: an original black cell inside a foreground-majority
#      3x3 neighborhood becomes a 3x3 patch of the opposite foreground color with a teal center.
#    - New behavior: a single foreground-colored cell may also appear inside the black background.
#      If the cleaned 3x3 neighborhood is uniquely background-majority, that cell is treated as
#      a reverse dent of the background region. The output paints a 3x3 black patch and then marks
#      the center teal. To make this visible, the generator places such dents near foreground edges
#      so the black patch bites into nearby foreground pixels.
#    - Small trace:
#        original logic: a lone red pixel in black background is cleanup noise and simply vanishes.
#        new logic: if its cleaned 3x3 still has black as the unique majority, that red pixel is a
#        background dent, so after cleanup the solver stamps black on the 3x3 and puts teal at the
#        center. The difference is therefore not only "the speck disappears", but "the background
#        claims a full 3x3 neighborhood around the speck and marks the center".
# b) Complexity calibration
#    - The variant must be visible in 2-4 examples, so background dents cannot be rare accidents.
#      The generator forces 2-5 background dents per instance.
#    - Background dents are restricted to full 3x3 neighborhoods with 1-3 nearby foreground cells:
#      this keeps the rule observable without flooding the grid with black patches.
#    - Ordinary foreground-vs-foreground boundary noise is retained so the cleanup stage remains
#      necessary, but background specks are no longer used as disposable noise because that would
#      make "some fg-in-bg cells matter and some do not" too hard to infer from few examples.
# c) Original assumptions that break under D8
#    - `_dent_centers` in the base task iterates only over original black cells. That silently
#      assumes background never hosts dents; this must be generalized.
#    - The base solver also assumes every detected dent stamps the opposite foreground color.
#      Under D8, a background-host dent instead stamps black.
#    - The cleanup pass itself is still valid, but now its result is used to detect dents in all
#      three host-region types, not only inside foreground.
# d) Placement legality
#    - All dent centers obey a Chebyshev separation of at least 3, so their 3x3 effects do not
#      overlap and oracle verification stays deterministic.
#    - Background dent candidates are interior background cells whose clean 3x3 is background-
#      dominant and includes some nearby foreground, so the inserted foreground speck will be
#      cleaned back to black before detection.
#    - Boundary-noise cells are kept at least one cell away from all dent centers so they cannot
#      perturb any dent's detection neighborhood.
# e) Visual observability
#    - Humans can now see three host situations: black holes inside color A, black holes inside
#      color B, and color specks inside black background. All three produce teal centers, but the
#      3x3 patch color follows the host region, not the inserted defect color.
#    - No rule uses raw color ids for ordering or branching; decisions are based only on local
#      neighborhood majorities visible in the grid.
# f) Visual cleanliness
#    - Background dents are forced near boundaries but still separated, so the black 3x3 patches
#      are legible rather than merging into a large blot.
#    - Pair-shaped dents remain only for foreground dents. Background dents stay single-cell so the
#      new signal is easy to isolate visually.
# g) Color-change inferability
#    - Only two foreground colors plus black/teal appear.
#    - Foreground dents still swap to the other foreground color; background dents use black. The
#      host-region explanation is therefore consistent across all cases and can be inferred from a
#      few examples.
# 9) New solver assumptions audited against the generator
#    - The solver assumes a true dent cell becomes black after cleanup. The generator guarantees
#      this for both foreground dents and background dents.
#    - The solver assumes the dominant host in the cleaned 3x3 is unique and has at least 5 cells.
#      The generator only places dents where that is true.
#    - The solver ignores original foreground-in-foreground contamination. The generator keeps such
#      noise away from the dent pattern and makes cleanup restore it to its host region.

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
    Remove one-cell specks/protrusions before detecting dents in any host region.

    Two sweeps are still enough: first remove the isolated corruption, then let the
    local majority stabilize around the corrected neighborhood.
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
    Return (r, c, host, stamp_color) for all dents.

    Every true dent cell becomes black after cleanup. Then:
    - black-in-foreground dents have a foreground host and stamp the opposite fg color
    - fg-in-background dents have a background host and stamp black
    """
    h, w = original.shape
    centers: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(clean[r, c]) != int(bg):
                continue
            host, _counts = _dominant_host(_sub3x3(clean, r, c), colors, bg)
            if host is None:
                continue

            v = int(original[r, c])
            if int(host) == int(bg):
                if v in colors:
                    centers.append((r, c, int(bg), int(bg)))
            else:
                if v != int(bg):
                    continue
                other = int(colors[1]) if int(host) == int(colors[0]) else int(colors[0])
                centers.append((r, c, int(host), other))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn dents in foreground and background regions into marked 3x3 patches.

    Input structure:
    - Black background with two large foreground colors.
    - The foreground regions contain black holes and a few one-cell cross-color
      boundary contaminants.
    - The background region can also contain isolated foreground-colored reverse
      dents near region edges.

    Rule:
    - First clean the two foreground colors locally: isolated fg specks in black
      revert to black, and one-cell boundary contamination reverts to its host.
    - Then inspect every cell that becomes black after cleanup. If the cleaned 3x3
      neighborhood is uniquely dominated by one foreground color and the original
      cell was black, stamp the opposite foreground color on its 3x3 neighborhood.
    - If the cleaned 3x3 neighborhood is uniquely dominated by black and the
      original cell was foreground-colored, stamp black on its 3x3 neighborhood.
      Every detected dent center is finally recolored teal (8).

    Key insight:
    - The transformation is host-region driven after cleanup. A defect cell marks
      where the local majority region was interrupted, and the output paints a 3x3
      patch in that host's counterpart: the other foreground for fg hosts, black
      for background hosts.
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
FG_DENT_COUNT_MIN, FG_DENT_COUNT_MAX = 10, 18
BG_DENT_COUNT_MIN, BG_DENT_COUNT_MAX = 2, 5
FG_DENT_MIN_HOST_3x3 = 5
BG_DENT_MIN_BG_3x3 = 6
BG_DENT_MAX_FG_3x3 = 3
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 5, 14
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles (simple occlusion, painter's algorithm)."""
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
    """After setting (r,c) to black, the solver must detect it as a foreground dent."""
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


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    blocked: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Select n centers with global Chebyshev separation."""
    blocked = list(blocked or [])
    pool = candidates[:]
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for r, c in pool:
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in blocked + chosen):
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
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    options = _bg_dent_color_options(clean, r, c, colors, Color.BLACK)
    assert options, "background dent must have at least one safe inserted color"
    best = max(options, key=lambda color: counts[int(color)])
    best_options = [int(color) for color in options if counts[int(color)] == counts[int(best)]]
    return int(rng.choice(best_options))


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
    """Check that changing (r,c) to `other` won't cause local cleanup cascades nearby."""
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
    """Place cross-color contamination at region-region boundaries, away from dents."""
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

    n_fg = rng.randint(FG_DENT_COUNT_MIN, FG_DENT_COUNT_MAX)
    n_bg = rng.randint(BG_DENT_COUNT_MIN, BG_DENT_COUNT_MAX)
    n_a, n_b = n_fg // 2, n_fg - n_fg // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

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

    if len(valid_fg[int(color_a)]) < n_a + 2 or len(valid_fg[int(color_b)]) < n_b + 2:
        return None
    if len(bg_options) < n_bg:
        return None

    dents_a = _pick_dent_centers(valid_fg[int(color_a)], n_a, rng)
    if dents_a is None:
        return None
    dents_b = _pick_dent_centers(valid_fg[int(color_b)], n_b, rng, blocked=dents_a)
    if dents_b is None:
        return None
    dents_bg = _pick_dent_centers(list(bg_options), n_bg, rng, blocked=dents_a + dents_b)
    if dents_bg is None:
        return None

    base_centers = dents_a + dents_b + dents_bg
    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, base_centers, rng)
    extra_a = [cell for cells in groups_a for cell in cells[1:]]
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, base_centers + extra_a, rng)

    output = _stamp_dents(clean, groups_a, groups_b, dents_bg, color_a, color_b)

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = Color.BLACK
    for r, c in dents_bg:
        inp[r, c] = _choose_bg_dent_color(clean, r, c, colors, rng)

    dent_cells = [cell for cells in groups_a + groups_b for cell in cells] + dents_bg
    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    _add_boundary_noise(inp, clean, n_noise, color_a, color_b, dent_cells, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None

    bg_event_count = sum(1 for _r, _c, host, _stamp in _dent_events(inp, _clean_regions(inp, colors, Color.BLACK), colors, Color.BLACK) if host == Color.BLACK)
    if bg_event_count < n_bg:
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
    raise RuntimeError("generate_input: failed after %d attempts" % MAX_GENERATION_ATTEMPTS)
