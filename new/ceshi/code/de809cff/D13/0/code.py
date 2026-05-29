# Variant of de809cff
# Dimension: D13 — 两种前景色之间的对调映射关系 (meta)
# Change: 固定双色对调：区域A的凹痕用色B填充补丁，区域B的凹痕用色A填充补丁（简单二元交换）... → 颜色映射由图例区域定义

from __future__ import annotations

# Design analysis
# - Concrete appearance:
#   The original scene stays intact: two large foreground regions sit on black,
#   black dents are punched into them, and a few single-cell noise pixels are
#   scattered in background or on the region boundary. The D13 change adds one
#   isolated 2x2 legend block on a grid edge. The outer edge row/column lists
#   the two host colors; the adjacent inner row/column lists the colors to use
#   for that host's dent patch. Because there are only two hosts, a fixed 2x2
#   legend is enough and keeps the symbolic step visually obvious.
# - Behavior trace:
#   Original logic: if a dent belongs to region A, stamp color B; if it belongs
#   to region B, stamp color A. New logic: read the legend first. Example:
#   top-edge legend outer row [A, B], inner row [C, A] means A→C and B→A. A's
#   dents now get C patches, B's dents now get A patches. The teal center mark
#   still identifies the original black dent location.
# - Complexity calibration:
#   The legend is small and repeated only once, so it does not overwhelm the
#   scene, but it is also not hidden: it sits on an edge and is surrounded by a
#   one-cell black moat. Mapping values are chosen from {A, B, C}, where C is a
#   third non-black, non-teal color. This keeps the rule space small enough for
#   ARC while making the meta-rule visible in 2-4 examples.
# - Old assumptions that break under D13:
#   `_clean_foreground` used to erase every non-main color to black. That would
#   destroy legend cells when a third color is used, so legend cells must be
#   preserved and excluded from cleanup.
#   `main()` used to hard-code "other foreground color" as the patch color. That
#   must become a lookup from host color into the decoded legend mapping.
#   Patch stamping used to be free to overwrite any cell in the 3x3 window. With
#   a legend now present, generator keeps dents away from the reserved legend
#   zone, and solver also avoids overwriting legend cells as a safety guard.
# - Placement legality:
#   Generator reserves the legend block plus its one-cell moat before painting
#   the large rectangles. Dent centers, background specks, and boundary-noise
#   cells are all forbidden from that reserved zone; dent centers are also
#   filtered so their 3x3 patch cannot touch it.
# - Visual observability / cleanliness:
#   The legend side and key order vary, but the structure is stable: edge-facing
#   keys, inward-facing mapped colors, black moat around the block. This makes
#   the symbol-decoding step easy to spot and keeps the output uncluttered.
# - New solver assumptions and generator guarantees:
#   Solver assumes there is exactly one isolated 2x2 edge legend whose outer
#   edge cells are the two dominant region colors. Generator guarantees that by
#   placing exactly one such block, keeping it isolated by black, and ensuring
#   the two large regions remain far larger than the legend colors by count.

import random
from collections import Counter as _Counter

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


def _legend_masks(shape: tuple[int, int], side: str, start: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    legend = np.zeros((h, w), dtype=bool)
    if side == "top":
        r0, r1, c0, c1 = 0, 2, start, start + 2
    elif side == "bottom":
        r0, r1, c0, c1 = h - 2, h, start, start + 2
    elif side == "left":
        r0, r1, c0, c1 = start, start + 2, 0, 2
    else:
        r0, r1, c0, c1 = start, start + 2, w - 2, w

    legend[r0:r1, c0:c1] = True
    reserved = np.zeros((h, w), dtype=bool)
    reserved[max(0, r0 - 1) : min(h, r1 + 1), max(0, c0 - 1) : min(w, c1 + 1)] = True
    return legend, reserved


def _legend_array(side: str, keys: tuple[int, int], fills: tuple[int, int]) -> np.ndarray:
    a, b = map(int, keys)
    x, y = map(int, fills)
    if side == "top":
        return np.array([[a, b], [x, y]], dtype=int)
    if side == "bottom":
        return np.array([[x, y], [a, b]], dtype=int)
    if side == "left":
        return np.array([[a, x], [b, y]], dtype=int)
    return np.array([[x, a], [y, b]], dtype=int)


def _paint_legend(canvas: np.ndarray, side: str, start: int, legend_block_2x2: np.ndarray) -> None:
    if side == "top":
        canvas[0:2, start : start + 2] = legend_block_2x2
    elif side == "bottom":
        canvas[-2:, start : start + 2] = legend_block_2x2
    elif side == "left":
        canvas[start : start + 2, 0:2] = legend_block_2x2
    else:
        canvas[start : start + 2, -2:] = legend_block_2x2


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


def _palette_mapping(g: np.ndarray, colors: tuple[int, int], bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = g.shape
    a, b = map(int, colors)
    color_set = {a, b}
    candidates: list[tuple[dict[int, int], np.ndarray]] = []

    if w >= 2:
        for c0 in range(w - 1):
            legend_mask, _reserved = _legend_masks((h, w), "top", c0)
            block = g[0:2, c0 : c0 + 2]
            outer = [int(x) for x in block[0].tolist()]
            inner = [int(x) for x in block[1].tolist()]
            if set(outer) == color_set and len(set(outer)) == 2 and all(v != bg for v in inner):
                if np.all(block != int(bg)) and _surrounding_ring_is_bg(g, legend_mask, bg):
                    candidates.append(({outer[0]: inner[0], outer[1]: inner[1]}, legend_mask))

        for c0 in range(w - 1):
            legend_mask, _reserved = _legend_masks((h, w), "bottom", c0)
            block = g[h - 2 : h, c0 : c0 + 2]
            outer = [int(x) for x in block[1].tolist()]
            inner = [int(x) for x in block[0].tolist()]
            if set(outer) == color_set and len(set(outer)) == 2 and all(v != bg for v in inner):
                if np.all(block != int(bg)) and _surrounding_ring_is_bg(g, legend_mask, bg):
                    candidates.append(({outer[0]: inner[0], outer[1]: inner[1]}, legend_mask))

    if h >= 2:
        for r0 in range(h - 1):
            legend_mask, _reserved = _legend_masks((h, w), "left", r0)
            block = g[r0 : r0 + 2, 0:2]
            outer = [int(x) for x in block[:, 0].tolist()]
            inner = [int(x) for x in block[:, 1].tolist()]
            if set(outer) == color_set and len(set(outer)) == 2 and all(v != bg for v in inner):
                if np.all(block != int(bg)) and _surrounding_ring_is_bg(g, legend_mask, bg):
                    candidates.append(({outer[0]: inner[0], outer[1]: inner[1]}, legend_mask))

        for r0 in range(h - 1):
            legend_mask, _reserved = _legend_masks((h, w), "right", r0)
            block = g[r0 : r0 + 2, w - 2 : w]
            outer = [int(x) for x in block[:, 1].tolist()]
            inner = [int(x) for x in block[:, 0].tolist()]
            if set(outer) == color_set and len(set(outer)) == 2 and all(v != bg for v in inner):
                if np.all(block != int(bg)) and _surrounding_ring_is_bg(g, legend_mask, bg):
                    candidates.append(({outer[0]: inner[0], outer[1]: inner[1]}, legend_mask))

    assert len(candidates) == 1, "expected exactly one edge legend"
    mapping, legend_mask = candidates[0]
    assert set(mapping) == color_set, "legend keys must match the two main colors"
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


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int, legend_mask: np.ndarray) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting black dents.

    Legend cells are preserved verbatim; the cleanup only applies to the two
    large regions and treats the legend as off-limits structure.
    """
    h, w = g.shape
    a, b = map(int, colors)
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                if legend_mask[r, c]:
                    nxt[r, c] = cur[r, c]
                    continue

                v = int(cur[r, c])
                if v not in colors:
                    nxt[r, c] = bg
                    continue

                rs = slice(max(0, r - 1), min(h, r + 2))
                cs = slice(max(0, c - 1), min(w, c + 2))
                sub = cur[rs, cs]
                sub_legend = legend_mask[rs, cs]
                own = int(np.sum((sub == v) & (~sub_legend)))
                other = b if v == a else a
                other_count = int(np.sum((sub == other) & (~sub_legend)))
                bg_count = int(np.sum((sub == bg) & (~sub_legend)))

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
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
    legend_mask: np.ndarray,
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where((original == int(bg)) & (~legend_mask)), strict=False):
        rr = int(r)
        cc = int(c)
        rs = slice(max(0, rr - 1), min(h, rr + 2))
        cs = slice(max(0, cc - 1), min(w, cc + 2))
        sub = clean[rs, cs]
        sub_legend = legend_mask[rs, cs]
        counts = {
            int(color): int(np.sum((sub == int(color)) & (~sub_legend)))
            for color in colors
        }
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _paint_patch(out: np.ndarray, legend_mask: np.ndarray, r: int, c: int, color: int) -> None:
    h, w = out.shape
    rs = slice(max(0, r - 1), min(h, r + 2))
    cs = slice(max(0, c - 1), min(w, c + 2))
    patch = out[rs, cs].copy()
    writable = ~legend_mask[rs, cs]
    patch[writable] = int(color)
    out[rs, cs] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into mapped patches directed by an edge legend.

    Input structure:
    - Black background with two large foreground colors and a small isolated 2x2 legend on one grid edge.
    - The legend's outer edge cells show the two host-region colors; the adjacent inner cells show the patch color assigned to each host.
    - The large regions still contain black dents plus a few one-cell color specks or protrusions near boundaries.

    Rule:
    - Read the edge legend as a host-color → patch-color mapping, then keep that legend unchanged.
    - Locally clean the two large regions: isolated specks become black, and one-cell cross-color protrusions revert to the surrounding region color.
    - Each original black dent whose cleaned 3x3 neighborhood is mostly one foreground color becomes the center of a 3x3 patch colored by that host's legend mapping, with the center recolored teal (8).

    Key insight: The dent detection step is the same local-majority repair as before, but the patch color is no longer "the other region"; it must be decoded from the edge legend first.
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

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: legend ---
LEGEND_SIDES = ("top", "bottom", "left", "right")
LEGEND_USE_THIRD_PROB = 0.72

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300


def _build_regions(
    N: int,
    color_a: int,
    color_b: int,
    reserved_mask: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
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

    grid[reserved_mask] = 0

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
    if grid[r, c] != host:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == other))


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


def _pick_dent_centers(candidates, n, rng):
    """Select n dent centers from candidates with Chebyshev distance >= DENT_PATCH_SEPARATION."""
    rng.shuffle(candidates)
    chosen = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _expand_to_pairs(dents, host, other, clean, all_dents, reserved_mask, rng):
    """Optionally expand single-cell dents into 2-cell adjacent pairs (~13% chance each)."""
    h, w = clean.shape
    result = []
    for r, c in dents:
        cells = [(r, c)]
        if rng.random() < DENT_PAIR_PROB:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if reserved_mask[nr, nc] or not _patch_clear_of_reserved(reserved_mask, nr, nc):
                    continue
                if clean[nr, nc] != host or not _is_safe_dent_center(clean, nr, nc, host, other):
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
    region_clean: np.ndarray,
    legend_mask: np.ndarray,
    legend_block: np.ndarray,
    dent_groups_a,
    dent_groups_b,
    color_a: int,
    color_b: int,
    mapping: dict[int, int],
) -> np.ndarray:
    """Build output by stamping mapped 3x3 patches and marking teal at dent centers."""
    h, w = region_clean.shape
    output = region_clean.copy()
    output[legend_mask] = legend_block[legend_mask]
    for dent_groups, host in [(dent_groups_a, color_a), (dent_groups_b, color_b)]:
        stamp_color = int(mapping[int(host)])
        for cells in dent_groups:
            for r, c in cells:
                _paint_patch(output, legend_mask, r, c, stamp_color)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, reserved_mask, rng):
    """Place isolated fg-color specks in background area."""
    bg_candidates = list(zip(*np.where((clean == 0) & (~reserved_mask)), strict=False))
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


def _neighbor_cleanup_safe(inp, r, c, host, other):
    """Check that changing (r,c) to `other` won't cause the solver to misclassify any neighbor."""
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or inp[nr, nc] != host:
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


def _add_boundary_noise(inp, clean, n_at_boundary, color_a, color_b, reserved_mask, rng):
    """Place cross-color contamination at region-region boundaries."""
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            if reserved_mask[r, c]:
                continue
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
        if inp[r, c] != host:
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


def _patch_clear_of_reserved(reserved_mask: np.ndarray, r: int, c: int) -> bool:
    return not bool(np.any(reserved_mask[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]))


def _dent_guard_mask(shape: tuple[int, int], dent_groups) -> np.ndarray:
    h, w = shape
    guard = np.zeros((h, w), dtype=bool)
    for cells in dent_groups:
        for r, c in cells:
            guard[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return guard


def _choose_mapping(color_a: int, color_b: int, rng: random.Random) -> tuple[dict[int, int], int]:
    third_pool = [c for c in Color.NOT_BLACK if c not in (Color.TEAL, color_a, color_b)]
    third = int(rng.choice(third_pool))
    fill_choices = [color_a, color_b, third]

    while True:
        if rng.random() < LEGEND_USE_THIRD_PROB:
            use_a_third = rng.random() < 0.5
            if use_a_third:
                fill_a = third
                fill_b = int(rng.choice([color_a, color_b, third]))
            else:
                fill_b = third
                fill_a = int(rng.choice([color_a, color_b, third]))
        else:
            fill_a = int(rng.choice(fill_choices))
            fill_b = int(rng.choice(fill_choices))

        if (fill_a, fill_b) == (color_a, color_b):
            continue
        if (fill_a, fill_b) == (color_b, color_a) and rng.random() < 0.7:
            continue
        return {int(color_a): fill_a, int(color_b): fill_b}, third


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    side = rng.choice(LEGEND_SIDES)
    start = rng.randint(0, N - 2)
    legend_mask, reserved_mask = _legend_masks((N, N), side, start)
    mapping, _third = _choose_mapping(color_a, color_b, rng)

    key_order = [color_a, color_b]
    rng.shuffle(key_order)
    fill_order = [mapping[key_order[0]], mapping[key_order[1]]]
    legend_block = np.zeros((N, N), dtype=int)
    _paint_legend(
        legend_block,
        side,
        start,
        _legend_array(side, (key_order[0], key_order[1]), (fill_order[0], fill_order[1])),
    )

    clean = _build_regions(N, color_a, color_b, reserved_mask, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    h, w = clean.shape
    valid = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            if reserved_mask[r, c] or not _patch_clear_of_reserved(reserved_mask, r, c):
                continue
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[color_b].append((r, c))

    if len(valid[color_a]) < n_a + 2 or len(valid[color_b]) < n_b + 2:
        return None

    dents_a = _pick_dent_centers(valid[color_a], n_a, rng)
    dents_b = _pick_dent_centers(valid[color_b], n_b, rng)
    if dents_a is None or dents_b is None:
        return None

    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, dents_a, reserved_mask, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, dents_b, reserved_mask, rng)
    noise_blocked = reserved_mask | _dent_guard_mask(clean.shape, groups_a + groups_b)

    output = _stamp_dents(
        clean,
        legend_mask,
        legend_block,
        groups_a,
        groups_b,
        color_a,
        color_b,
        mapping,
    )

    inp = clean.copy()
    inp[legend_mask] = legend_block[legend_mask]
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, noise_blocked, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, noise_blocked, rng)

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
