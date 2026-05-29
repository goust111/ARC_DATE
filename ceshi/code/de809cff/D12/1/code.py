# Variant of de809cff
# Dimension: D12 — 凹痕形状：单像素 vs 多像素 (src)
# Change: 凹痕主要为单个黑像素（1x1），偶尔出现相邻的2个黑像素... → 凹痕形状编码补丁方向

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis
# a) Concrete behavior trace
# - The region-cleaning phase stays unchanged: it removes isolated foreground specks
#   in the black background and fixes one-cell cross-color protrusions at region
#   boundaries.
# - The variant only changes the interpretation of black dents after cleaning.
#   Instead of handling each black pixel independently, the solver first groups
#   dent-like black cells into 4-connected components.
# - Accepted dent components:
#   * size 1: a single black cell -> standard 3x3 opposite-color square, center teal.
#   * size 2 in one row: cells (r,c),(r,c+1) -> 1x3 opposite-color stripe at
#     (r,c-1:c+1), then the two original dent cells become teal.
#   * size 2 in one column: cells (r,c),(r+1,c) -> 3x1 opposite-color stripe at
#     (r-1:r+1,c), then the two original dent cells become teal.
# - Small trace:
#   original rule on a horizontal pair blackened from [A A] would stamp two overlapping
#   3x3 squares, producing a bulky 3x4/3x3-like blob. The new rule groups that pair
#   first and stamps only a thin 3-cell horizontal bar, so the dent shape controls
#   the patch geometry.
# b) Complexity calibration
# - Only three dent classes exist: single, horizontal pair, vertical pair.
#   That is enough variation to make the new rule visible without introducing a large
#   case split that would be hard to infer from 2-4 examples.
# - Pair dents are still a minority, but each generated grid contains both pair
#   orientations somewhere so the shape-to-shape mapping is observable.
# c) Old implicit assumptions that must change
# - `_dent_centers` assumed every relevant black cell could be solved independently.
#   That breaks on 2-cell dents because two adjacent dent cells should no longer create
#   two overlapping 3x3 patches.
# - The old stamping loop unconditionally wrote a 3x3 square for every dent cell.
#   In the variant, the write shape depends on the dent component size/orientation.
# - Recoloring teal after stamping is still valid, but now a pair contributes two teal
#   cells inside one shared stripe rather than two separate square patches.
# d) Placement legality
# - Pair candidates are generated only when both dent cells are safe host cells and the
#   stripe's extra cap cell (left for horizontal, above for vertical) exists inside the
#   host region. This prevents clipped or ambiguous pair outputs from dominating.
# - Different dent groups obey a Chebyshev separation of at least 3 between all black
#   dent cells, keeping components disjoint and preventing patch overlap.
# e) Visual observability
# - Singles still create 3x3 blocks, while pairs create thin 3-cell bars. On the same
#   grid, that contrast is obvious by eye and cannot be mistaken for the original task's
#   "every dent makes a square" rule.
# - The rule uses only visible structure: number of black cells and whether the pair is
#   horizontal or vertical. No color-id ordering or hidden metadata is involved.
# f) Visual cleanliness
# - Pair bars are placed with an interior cap cell, so they usually appear as full
#   3-cell stripes rather than clipped fragments on the border.
# - Global spacing keeps teal markers and opposite-color patches from piling up.
# g) Color-change inference
# - Color behavior stays simple: every dent patch still uses the opposite foreground
#   color, and teal still marks the original black cells. Only the patch geometry changes.
# h) New solver assumptions and generator guarantees
# - Solver assumption: any multi-cell dent that should be transformed must appear as a
#   4-connected candidate component of size exactly 2 with a uniform host color.
#   Generator guarantee: it creates only single dents and 2-cell orthogonal pairs.
# - Solver assumption: the top-left-most cell anchors the odd-length stripe for a pair.
#   Generator guarantee: horizontal pairs always have a host cell to their left and
#   vertical pairs always have a host cell above, so the anchored stripe is visible.


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


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _dent_host(original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int, r: int, c: int) -> int | None:
    if int(original[r, c]) != int(bg):
        return None
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    host = max(colors, key=lambda color: counts[int(color)])
    other = colors[1] if int(host) == int(colors[0]) else colors[0]
    if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
        return int(host)
    return None


def _group_shape(cells: list[tuple[int, int]]) -> str | None:
    if len(cells) == 1:
        return "single"
    if len(cells) != 2:
        return None
    (r0, c0), (r1, c1) = sorted(cells)
    if r0 == r1 and abs(c0 - c1) == 1:
        return "horizontal"
    if c0 == c1 and abs(r0 - r1) == 1:
        return "vertical"
    return None


def _dent_groups(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[list[tuple[int, int]], int, str]]:
    h, w = original.shape
    candidates: dict[tuple[int, int], int] = {}
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        host = _dent_host(original, clean, colors, bg, rr, cc)
        if host is not None:
            candidates[(rr, cc)] = host

    groups: list[tuple[list[tuple[int, int]], int, str]] = []
    seen: set[tuple[int, int]] = set()
    for start in sorted(candidates):
        if start in seen:
            continue
        stack = [start]
        comp: list[tuple[int, int]] = []
        hosts: set[int] = set()
        while stack:
            cell = stack.pop()
            if cell in seen:
                continue
            seen.add(cell)
            comp.append(cell)
            hosts.add(int(candidates[cell]))
            r, c = cell
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in candidates and (nr, nc) not in seen:
                    stack.append((nr, nc))

        shape = _group_shape(comp)
        if shape is None or len(hosts) != 1:
            continue
        groups.append((sorted(comp), next(iter(hosts)), shape))
    return groups


def _stamp_group(out: np.ndarray, cells: list[tuple[int, int]], shape: str, color: int) -> None:
    h, w = out.shape
    if shape == "single":
        r, c = cells[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(color)
        return
    if shape == "horizontal":
        r = cells[0][0]
        c0 = min(c for _r, c in cells)
        for cc in range(c0 - 1, c0 + 2):
            if 0 <= cc < w:
                out[r, cc] = int(color)
        return
    if shape == "vertical":
        c = cells[0][1]
        r0 = min(r for r, _c in cells)
        for rr in range(r0 - 1, r0 + 2):
            if 0 <= rr < h:
                out[rr, c] = int(color)
        return
    raise ValueError(f"unknown dent shape: {shape}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents inside two colored regions into marked opposite-color patches.

    Input structure:
    - Black background with two large foreground-color regions.
    - The regions contain black dents plus a small amount of one-cell foreground noise
      in the background and at the color boundary.

    Rule:
    - First clean the foreground regions locally, removing isolated specks and undoing
      one-cell cross-color protrusions.
    - Then find black dent components whose cleaned 3x3 neighborhoods are mostly one
      foreground color: a single-cell dent becomes a 3x3 opposite-color square, while
      a 2-cell horizontal/vertical dent becomes a 3-cell stripe in the same orientation.
    - Finally recolor every original dent cell itself to teal (8).

    Key insight:
    - Teal marks the original black defect locations, but the surrounding opposite-color
      patch is no longer uniform: the dent's geometry (single vs horizontal pair vs
      vertical pair) decides whether the patch is a square or an oriented stripe.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    groups = _dent_groups(g, clean, colors, bg)
    for cells, host, shape in groups:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        _stamp_group(out, cells, shape, int(other))

    for cells, _host, _shape in groups:
        for r, c in cells:
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
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
PAIR_COUNT_MIN, PAIR_COUNT_MAX = 2, 4

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
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


def _is_safe_dent_cell(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


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


def _all_far_enough(cells: list[tuple[int, int]], chosen: list[list[tuple[int, int]]]) -> bool:
    for group in chosen:
        for r, c in cells:
            for rr, cc in group:
                if max(abs(r - rr), abs(c - cc)) < DENT_PATCH_SEPARATION:
                    return False
    return True


def _enumerate_single_candidates(clean: np.ndarray, host: int, other: int) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    candidates: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if _is_safe_dent_cell(clean, r, c, host, other):
                candidates.append([(r, c)])
    return candidates


def _enumerate_pair_candidates(
    clean: np.ndarray, host: int, other: int, shape: str
) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    candidates: list[list[tuple[int, int]]] = []
    if shape == "horizontal":
        for r in range(h):
            for c in range(1, w - 1):
                cells = [(r, c), (r, c + 1)]
                cap = (r, c - 1)
                if any(int(clean[rr, cc]) != int(host) for rr, cc in cells + [cap]):
                    continue
                if all(_is_safe_dent_cell(clean, rr, cc, host, other) for rr, cc in cells):
                    candidates.append(cells)
        return candidates
    if shape == "vertical":
        for r in range(1, h - 1):
            for c in range(w):
                cells = [(r, c), (r + 1, c)]
                cap = (r - 1, c)
                if any(int(clean[rr, cc]) != int(host) for rr, cc in cells + [cap]):
                    continue
                if all(_is_safe_dent_cell(clean, rr, cc, host, other) for rr, cc in cells):
                    candidates.append(cells)
        return candidates
    raise ValueError(f"unknown pair shape: {shape}")


def _pick_from_candidates(
    candidates: list[list[tuple[int, int]]],
    n: int,
    rng: random.Random,
    chosen_global: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]] | None:
    pool = [list(cells) for cells in candidates]
    rng.shuffle(pool)
    picked: list[list[tuple[int, int]]] = []
    for cells in pool:
        if _all_far_enough(cells, chosen_global + picked):
            picked.append(cells)
            if len(picked) == n:
                return picked
    return None if n > 0 else []


def _stamp_dent_groups(
    clean: np.ndarray,
    dent_groups_a: list[tuple[list[tuple[int, int]], str]],
    dent_groups_b: list[tuple[list[tuple[int, int]], str]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    output = clean.copy()
    for dent_groups, stamp_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells, shape in dent_groups:
            _stamp_group(output, cells, shape, int(stamp_color))
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells, _shape in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
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
    """Check that changing (r,c) to `other` won't cause the solver to misclassify any neighbor."""
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


def _plan_pair_targets(
    n_a: int, n_b: int, color_a: int, color_b: int, rng: random.Random
) -> tuple[dict[int, dict[str, int]], dict[int, int]]:
    total_pairs = rng.randint(PAIR_COUNT_MIN, min(PAIR_COUNT_MAX, n_a + n_b - 2))
    pair_targets = {
        color_a: {"horizontal": 0, "vertical": 0},
        color_b: {"horizontal": 0, "vertical": 0},
    }
    remaining = {color_a: n_a, color_b: n_b}

    shapes = ["horizontal", "vertical"]
    while len(shapes) < total_pairs:
        shapes.append(rng.choice(["horizontal", "vertical"]))
    rng.shuffle(shapes)

    for shape in shapes:
        choices = [color for color in (color_a, color_b) if remaining[color] > 0]
        if not choices:
            raise ValueError("no remaining dent slots for pair targets")
        choices.sort(key=lambda color: remaining[color], reverse=True)
        top_remaining = remaining[choices[0]]
        choices = [color for color in choices if remaining[color] == top_remaining]
        chosen_color = rng.choice(choices)
        pair_targets[chosen_color][shape] += 1
        remaining[chosen_color] -= 1

    return pair_targets, remaining


def _choose_groups_for_color(
    clean: np.ndarray,
    host: int,
    other: int,
    pair_targets: dict[str, int],
    single_count: int,
    rng: random.Random,
    chosen_global: list[list[tuple[int, int]]],
) -> list[tuple[list[tuple[int, int]], str]] | None:
    groups: list[tuple[list[tuple[int, int]], str]] = []
    picked_h = _pick_from_candidates(
        _enumerate_pair_candidates(clean, host, other, "horizontal"),
        pair_targets["horizontal"],
        rng,
        chosen_global,
    )
    if picked_h is None:
        return None
    groups.extend((cells, "horizontal") for cells in picked_h)

    picked_v = _pick_from_candidates(
        _enumerate_pair_candidates(clean, host, other, "vertical"),
        pair_targets["vertical"],
        rng,
        chosen_global + [cells for cells, _shape in groups],
    )
    if picked_v is None:
        return None
    groups.extend((cells, "vertical") for cells in picked_v)

    picked_single = _pick_from_candidates(
        _enumerate_single_candidates(clean, host, other),
        single_count,
        rng,
        chosen_global + [cells for cells, _shape in groups],
    )
    if picked_single is None:
        return None
    groups.extend((cells, "single") for cells in picked_single)
    return groups


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

    pair_targets, single_targets = _plan_pair_targets(n_a, n_b, color_a, color_b, rng)

    groups_a = _choose_groups_for_color(
        clean,
        color_a,
        color_b,
        pair_targets[color_a],
        single_targets[color_a],
        rng,
        [],
    )
    if groups_a is None:
        return None

    groups_b = _choose_groups_for_color(
        clean,
        color_b,
        color_a,
        pair_targets[color_b],
        single_targets[color_b],
        rng,
        [cells for cells, _shape in groups_a],
    )
    if groups_b is None:
        return None

    output = _stamp_dent_groups(clean, groups_a, groups_b, color_a, color_b)

    inp = clean.copy()
    for cells, _shape in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = 0

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
    raise RuntimeError("generate_input: failed after %d attempts" % MAX_GENERATION_ATTEMPTS)
