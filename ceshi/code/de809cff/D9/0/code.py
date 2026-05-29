# Variant of de809cff
# Dimension: D9 — 3x3补丁与区域边界的交互 (meta)
# Change: 3x3补丁完全在原始凹痕的3x3邻域内生成，可能覆盖同区域的前景像素，边界凹痕的补丁可能延伸到背景或另一区域... → 补丁被裁剪到所属区域边界内

from __future__ import annotations

# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior:
#    Original rule: once a black dent center is found, the solver paints the entire
#    clipped-to-grid 3x3 box with the opposite foreground color, even if some cells
#    of that 3x3 box belong to background or the other foreground region.
#    New rule: the same 3x3 box is proposed first, but only those cells that still
#    belong to the dent's host region are recolored; any cell crossing the host
#    region boundary is left unchanged. Example around a boundary dent:
#      host host host        other other other        other other teal
#      host dent bg     ->   other other other   vs   other teal  bg
#      host bg   other       other other other        other bg    other
#    Here the old task would overwrite the bg/other cells on the right/bottom edge,
#    while the new task keeps them intact and yields an irregular L-shaped patch.
#    The changed stage is patch write-back only; cleanup and dent detection stay the
#    same.
# b) Complexity calibration:
#    The task still has only two large regions, one cleanup phase, and local dent
#    reasoning. Complexity is increased only by requiring the solver to infer the
#    dent's host-region mask before recoloring the 3x3 neighborhood. The generator
#    therefore enforces several boundary dents per sample so the clipping effect is
#    repeated and visible, but keeps interior dents too so the irregular-vs-square
#    contrast is easy to see.
# c) Hidden assumptions reviewed:
#    - `out[r-1:r+2, c-1:c+2] = other` assumed every in-bounds cell in the 3x3 box
#      should flip. That is false in D9; we now need a host-region mask.
#    - Using the cleaned grid directly as that mask is also unsafe because dent cells
#      remain black after cleanup, so paired dents would incorrectly punch holes in
#      each other's patches. The solver reconstructs the host region by filling every
#      detected dent center back with its host color before clipping.
#    - The dent detection logic still assumes the host color is the strict local
#      majority in the cleaned 3x3 neighborhood; the generator preserves that.
# d) Placement legality:
#    Generator dents are chosen from cells that remain valid under the original local
#    host-majority test, with Chebyshev separation between clusters. A configurable
#    quota is drawn from host-majority cells whose 3x3 neighborhood also touches
#    non-host cells inside the grid, guaranteeing real boundary clipping without
#    out-of-bounds artifacts being the only difference.
# e) Visual observability:
#    Each sample contains multiple clipped dents across both colors, so some output
#    marks stay full 3x3 blocks while others become truncated wedges or L-shapes.
#    This repeated contrast is visible from 2-3 examples without relying on color id
#    ordering or hidden metadata.
# f) Visual cleanliness:
#    The same spacing constraints as the original task are retained, so irregular
#    patches remain separated except for optional 2-cell dent pairs that already form
#    a coherent local motif. Clipping only removes spillover; it does not add new
#    overlay layers.
# g) Color-rule inferability:
#    Colors are unchanged from the original task: the patch uses the opposite major
#    foreground color and the dent center becomes teal. D9 changes only patch shape,
#    not the color mapping, so the color rule remains easy to infer.

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


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


def _reconstruct_regions(clean: np.ndarray, centers: list[tuple[int, int, int]]) -> np.ndarray:
    """Fill dent centers back with their host colors to recover the pre-dent region mask."""
    region_mask = clean.copy()
    for r, c, host in centers:
        region_mask[r, c] = int(host)
    return region_mask


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
    Turn black dents in two colored regions into clipped opposite-color patches.

    Input structure:
    - Black background with two large foreground colors.
    - The foreground regions contain small black dents plus a few isolated color
      specks or one-cell boundary protrusions that should be cleaned first.

    Rule:
    - Locally clean the two foreground regions to remove isolated noise and revert
      one-cell cross-boundary contamination.
    - Each original black dent whose cleaned 3x3 neighborhood is mostly one
      foreground color becomes the center of an opposite-color 3x3 patch.
    - Unlike the base task, keep only the patch cells that lie inside that dent's
      host foreground region; any part of the 3x3 box crossing into background or
      the other region is left unchanged, and the dent center is then colored teal.

    Key insight: Reconstruct the dent's host region after cleanup, then use that
    region mask to clip each opposite-color 3x3 patch before marking its center.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    centers = _dent_centers(g, clean, colors, bg)
    region_mask = _reconstruct_regions(clean, centers)

    out = clean.copy()
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        _paint_clipped_patch(out, region_mask, r, c, int(host), int(other))

    for r, c, _host in centers:
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
DENT_PAIR_PROB = 0.13
CLIPPED_DENT_FRAC = 0.35
MIN_CLIPPED_DENTS_PER_COLOR = 2
MIN_INTERIOR_DENTS_PER_COLOR = 2
MIN_TOTAL_CLIP_LOSS = 8

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300
SELECT_RETRY_ATTEMPTS = 120


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    """Return the 3x3 neighborhood of (r,c), clipped to grid bounds."""
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


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


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """After setting (r,c) to black, the solver must detect it as a dent in host region."""
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _is_clipped_candidate(grid: np.ndarray, r: int, c: int, host: int) -> bool:
    """A dent is visually clipped only if its full 3x3 box is in-bounds and touches non-host cells."""
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    sub = _sub3x3(grid, r, c)
    return bool(np.any(sub != int(host)))


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    """Check whether placing `color` at (r,c) would complete a 2x2 block of that color."""
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
    preferred: list[tuple[int, int]] | None = None,
    preferred_target: int = 0,
) -> list[tuple[int, int]] | None:
    """Select n centers with spacing, forcing a minimum number from `preferred` if requested."""
    preferred = [] if preferred is None else list(preferred)
    candidate_set = set(candidates)
    preferred = [p for p in preferred if p in candidate_set]

    for _ in range(SELECT_RETRY_ATTEMPTS):
        chosen: list[tuple[int, int]] = []

        pref_order = preferred[:]
        rng.shuffle(pref_order)
        for r, c in pref_order:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) >= preferred_target:
                    break
        if len(chosen) < preferred_target:
            continue

        remaining = [p for p in candidates if p not in chosen]
        rng.shuffle(remaining)
        for r, c in remaining:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) >= n:
                    return chosen
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    all_dents: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    """Optionally expand single-cell dents into 2-cell adjacent pairs (~13% chance each)."""
    h, w = clean.shape
    result: list[list[tuple[int, int]]] = []
    for r, c in dents:
        cells = [(r, c)]
        if rng.random() < DENT_PAIR_PROB:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(clean[nr, nc]) != int(host) or not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if all(
                    max(abs(nr - r2), abs(nc - c2)) >= DENT_PATCH_SEPARATION
                    for r2, c2 in all_dents
                    if (r2, c2) != (r, c)
                ):
                    cells.append((nr, nc))
                    break
        result.append(cells)
    return result


def _stamp_dents(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    color_a: int,
    color_b: int,
    *,
    clipped: bool,
) -> np.ndarray:
    """Build output with either clipped or legacy unclipped 3x3 opposite-color patches."""
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, host, stamp_color in (
        (dent_groups_a, color_a, color_b),
        (dent_groups_b, color_b, color_a),
    ):
        for cells in dent_groups:
            for r, c in cells:
                if clipped:
                    for rr in range(max(0, r - 1), min(h, r + 2)):
                        for cc in range(max(0, c - 1), min(w, c + 2)):
                            if int(clean[rr, cc]) == int(host):
                                output[rr, cc] = int(stamp_color)
                else:
                    output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(
                        stamp_color
                    )
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _clip_metrics(clean: np.ndarray, dent_groups: list[list[tuple[int, int]]], host: int) -> tuple[int, int]:
    """Return (number of clipped dents, total spilled cells prevented by clipping)."""
    clipped_dents = 0
    total_loss = 0
    for cells in dent_groups:
        for r, c in cells:
            sub = _sub3x3(clean, r, c)
            loss = int(sub.size - np.sum(sub == int(host)))
            if loss > 0:
                clipped_dents += 1
                total_loss += loss
    return clipped_dents, total_loss


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
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
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
            if int(np.sum(_sub3x3(clean, r, c) == int(other))) >= 1:
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
        inp[r, c] = other
        placed += 1


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
    valid = {color_a: [], color_b: []}
    clipped = {color_a: [], color_b: []}
    interior = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[color_a].append((r, c))
                if _is_clipped_candidate(clean, r, c, color_a):
                    clipped[color_a].append((r, c))
                else:
                    interior[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[color_b].append((r, c))
                if _is_clipped_candidate(clean, r, c, color_b):
                    clipped[color_b].append((r, c))
                else:
                    interior[color_b].append((r, c))

    if len(valid[color_a]) < n_a + 2 or len(valid[color_b]) < n_b + 2:
        return None

    target_clip_a = max(MIN_CLIPPED_DENTS_PER_COLOR, int(round(n_a * CLIPPED_DENT_FRAC)))
    target_clip_b = max(MIN_CLIPPED_DENTS_PER_COLOR, int(round(n_b * CLIPPED_DENT_FRAC)))
    if len(clipped[color_a]) < target_clip_a or len(clipped[color_b]) < target_clip_b:
        return None
    if len(interior[color_a]) < MIN_INTERIOR_DENTS_PER_COLOR or len(interior[color_b]) < MIN_INTERIOR_DENTS_PER_COLOR:
        return None

    dents_a = _pick_dent_centers(valid[color_a], n_a, rng, clipped[color_a], target_clip_a)
    dents_b = _pick_dent_centers(valid[color_b], n_b, rng, clipped[color_b], target_clip_b)
    if dents_a is None or dents_b is None:
        return None

    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, dents_a, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, dents_b, rng)

    clip_dents_a, clip_loss_a = _clip_metrics(clean, groups_a, color_a)
    clip_dents_b, clip_loss_b = _clip_metrics(clean, groups_b, color_b)
    if clip_dents_a < target_clip_a or clip_dents_b < target_clip_b:
        return None
    if clip_loss_a + clip_loss_b < MIN_TOTAL_CLIP_LOSS:
        return None

    output = _stamp_dents(clean, groups_a, groups_b, color_a, color_b, clipped=True)
    legacy_output = _stamp_dents(clean, groups_a, groups_b, color_a, color_b, clipped=False)
    if np.array_equal(output, legacy_output):
        return None

    inp = clean.copy()
    for cells in groups_a + groups_b:
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
