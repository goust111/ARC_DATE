# Combo variant of de809cff (index 31)
# combo_spec: D12/0+D16/1
# Name: 凹痕对称分布+多像素连通凹痕

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np
from scipy.ndimage import label

from arc_utils import Color


# Design analysis for the fused variant
# a) Concrete behavior trace
# - Original task:
#   1. Clean isolated foreground specks and one-cell boundary contamination.
#   2. Treat every surviving black dent pixel independently.
#   3. Around each dent center, stamp a 3x3 opposite-color patch and recolor the
#      dent center itself to teal.
# - Fused variant:
#   1. The cleanup stage stays the same.
#   2. Black dents are no longer single pixels. They are 4-connected components of
#      size 2-4, such as dominoes, 3-lines, small L-shapes, or 2x2 blocks.
#   3. Candidate dent components are only accepted when they come in point-symmetric
#      pairs inside the same cleaned foreground region. The region center is taken
#      from that region's visible bbox center, so reflection is an exact 180-degree
#      coordinate map.
#   4. Each accepted dent component still yields one 3x3 opposite-color patch, but
#      the patch center is derived from the dent component's centroid after
#      quantizing it to a grid cell under the symmetry constraint. Every original
#      black dent cell becomes teal, not just one cell.
# - Small trace:
#   If a left-region dent is an L-shape at {(5,5),(6,5),(6,6)}, the original solver
#   would either miss the structure or stamp three overlapping 3x3 blocks. Here the
#   solver groups those three black cells as one dent component, confirms that its
#   180-degree reflected partner also exists in the same region, then stamps exactly
#   one 3x3 patch for each member of the pair and turns the full L-shape teal.
# b) Complexity calibration
# - The variant uses only one new structural degree of freedom in each sample:
#   small multi-cell dents plus one global pairing rule inside each region.
# - The dent-shape catalog is limited to a handful of 2-4-cell connected motifs, and
#   every sample repeats the same "paired around the region center" relation several
#   times. That keeps the rule visible without exploding the case split.
# c) Old implicit assumptions that break
# - "Each black dent cell is an independent event" is false. Adjacent black cells now
#   belong to the same object and must not create multiple overlapping patches.
# - "A 3x3 majority around one black pixel is enough to classify the dent" is also
#   false. The host must now be inferred from the ring around the whole black
#   component.
# - "Any black component that looks dent-like should be transformed" is too weak for
#   the fused task. A candidate must also have a point-symmetric partner in the same
#   region, otherwise it is ignored.
# d) Placement legality
# - The generator builds two large rectangular foreground regions with a shared seam,
#   then places every dent pair strictly inside one region with at least a one-cell
#   host-colored moat. This prevents dents from merging with the external background.
# - Pair members, their teal footprints, and their 3x3 patches are all spaced apart
#   so dents remain separate components and outputs stay readable.
# e) Visual observability
# - A human can see that dents are multi-cell black blobs, not isolated pixels.
# - A human can also see that those blobs occur in 180-degree pairs around the center
#   of each region. The resulting opposite-color patches and teal markers therefore
#   appear in symmetric pairs as well.
# - No hidden ordering by color id or count is used; every decision comes from visible
#   geometry.
# f) Visual cleanliness
# - The generator keeps all dent patches fully inside their host rectangles, so the
#   output is mostly clean 3x3 blocks instead of many clipped fragments.
# - Cross-color noise stays sparse and one-cell wide, which preserves the original
#   "cleanup first" cue without obscuring the dent-pair structure.
# g) Color-change inference
# - Color logic stays simple and repeated: dents inside color A become 3x3 blocks of
#   color B, and dents inside color B become 3x3 blocks of color A. Teal marks every
#   original black dent cell.
# h) New solver assumptions and generator guarantees
# - Solver assumption: every transformable dent is a black 4-connected component of
#   size 2-4 surrounded mostly by one cleaned region component.
#   Generator guarantee: it creates only such components and leaves a host-colored
#   moat around them.
# - Solver assumption: every transformable dent has an exact 180-degree partner in
#   the same region.
#   Generator guarantee: dent pairs are sampled by choosing one component and placing
#   the second by exact point reflection through the region bbox center.
# - Solver assumption: centroid quantization must preserve pair symmetry.
#   Generator guarantee: expected outputs use the same symmetry-constrained anchor
#   rule as the solver, so there is no training-time mismatch.


GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.82

DENT_PAIRS_PER_REGION_MIN = 2
DENT_PAIRS_PER_REGION_MAX = 4
DENT_COMPONENT_MIN_SIZE = 2
DENT_COMPONENT_MAX_SIZE = 4
DENT_CELL_SEPARATION = 3
DENT_PATCH_SEPARATION = 4

NOISE_COUNT_MIN, NOISE_COUNT_MAX = 5, 12
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.82, 0.95
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

MAX_GENERATION_ATTEMPTS = 300
PAIR_PLACEMENT_ATTEMPTS = 40

SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
)


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
                elif bg_count >= 6 and own <= 3:
                    nxt[r, c] = bg
                elif own <= 1:
                    nxt[r, c] = bg
                else:
                    nxt[r, c] = v
        cur = nxt

    return cur


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _region_data(
    clean: np.ndarray, colors: tuple[int, int]
) -> tuple[np.ndarray, dict[int, int], dict[int, tuple[int, int]], dict[int, tuple[int, int, int, int]]]:
    region_ids = np.zeros_like(clean, dtype=int)
    region_color: dict[int, int] = {}
    region_center: dict[int, tuple[int, int]] = {}
    region_bbox: dict[int, tuple[int, int, int, int]] = {}
    next_id = 1
    for color in colors:
        labels, n = label(clean == int(color))
        for k in range(1, int(n) + 1):
            mask = labels == k
            if not bool(np.any(mask)):
                continue
            region_ids[mask] = next_id
            region_color[next_id] = int(color)
            rows, cols = np.where(mask)
            r0 = int(rows.min())
            r1 = int(rows.max())
            c0 = int(cols.min())
            c1 = int(cols.max())
            region_center[next_id] = (r0 + r1, c0 + c1)
            region_bbox[next_id] = (r0, c0, r1, c1)
            next_id += 1
    return region_ids, region_color, region_center, region_bbox


def _ring_coords(cells: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cell_set = set(cells)
    ring: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cell_set:
                    ring.add((rr, cc))
    return ring


def _choose_axis_anchor(total: int, n: int) -> int:
    lo = total // n
    hi = (total + n - 1) // n
    if lo == hi:
        return int(lo)
    err_lo = abs(n * lo - total)
    err_hi = abs(n * hi - total)
    if err_lo <= err_hi:
        return int(lo)
    return int(hi)


def _pair_patch_centers(
    cells_a: tuple[tuple[int, int], ...],
    cells_b: tuple[tuple[int, int], ...],
    center_double: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    first, second = (cells_a, cells_b) if cells_a <= cells_b else (cells_b, cells_a)
    n = len(first)
    sum_r = sum(r for r, _c in first)
    sum_c = sum(c for _r, c in first)
    anchor_first = (_choose_axis_anchor(sum_r, n), _choose_axis_anchor(sum_c, n))
    anchor_second = (center_double[0] - anchor_first[0], center_double[1] - anchor_first[1])
    return (anchor_first, anchor_second) if first == cells_a else (anchor_second, anchor_first)


def _stamp_patch(out: np.ndarray, center: tuple[int, int], color: int) -> None:
    h, w = out.shape
    r, c = center
    out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(color)


def _is_candidate_dent_component(
    cells: tuple[tuple[int, int], ...],
    clean: np.ndarray,
    region_ids: np.ndarray,
) -> tuple[int, int] | None:
    if not (DENT_COMPONENT_MIN_SIZE <= len(cells) <= DENT_COMPONENT_MAX_SIZE):
        return None
    h, w = clean.shape
    if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
        return None
    ring = _ring_coords(cells, clean.shape)
    host_counts: dict[int, int] = {}
    bg_count = 0
    for rr, cc in ring:
        rid = int(region_ids[rr, cc])
        if rid == 0:
            bg_count += 1
        else:
            host_counts[rid] = host_counts.get(rid, 0) + 1
    if not host_counts:
        return None
    host_id = max(host_counts, key=host_counts.get)
    host_count = host_counts[host_id]
    other_count = sum(v for rid, v in host_counts.items() if rid != host_id)
    if bg_count > 0:
        return None
    if host_count < len(cells) + 4:
        return None
    if host_count <= other_count + 2:
        return None
    return int(host_id), int(host_count)


def _reflect_cells(cells: tuple[tuple[int, int], ...], center_double: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    sr, sc = center_double
    return tuple(sorted((sr - r, sc - c) for r, c in cells))


def _paired_dent_components(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
) -> list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], int, int]]:
    region_ids, region_color, region_center, _region_bbox = _region_data(clean, colors)
    labels, n = label(original == int(Color.BLACK))

    candidates: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}
    h, w = original.shape
    for k in range(1, int(n) + 1):
        rows, cols = np.where(labels == k)
        cells = tuple(sorted((int(r), int(c)) for r, c in zip(rows, cols, strict=False)))
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            continue
        info = _is_candidate_dent_component(cells, clean, region_ids)
        if info is not None:
            candidates[cells] = info

    pairs: list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], int, int]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for cells, (region_id, _host_count) in sorted(candidates.items()):
        if cells in seen:
            continue
        reflected = _reflect_cells(cells, region_center[region_id])
        if reflected == cells:
            continue
        partner = candidates.get(reflected)
        if partner is None:
            continue
        partner_region_id, _partner_host_count = partner
        if partner_region_id != region_id:
            continue
        seen.add(cells)
        seen.add(reflected)
        pairs.append((cells, reflected, region_id, region_color[region_id]))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn symmetric multi-cell black dents inside two colored regions into paired
    opposite-color patches.

    Input structure:
    - Black background with two large foreground-color regions.
    - Each region may contain several 2-4-cell black dent components plus sparse
      one-cell foreground noise in the background or on the seam between regions.
    - True dents occur in point-symmetric pairs within the same foreground region.

    Rule:
    - First clean the foreground locally, removing isolated color specks and undoing
      one-cell cross-color contamination at the boundary.
    - Next group black cells into 4-connected dent components. Keep only those small
      components whose surrounding cleaned ring belongs to one foreground region and
      whose 180-degree reflected partner exists around that region's center.
    - For each accepted component, stamp one 3x3 patch in the opposite foreground
      color using the component centroid as the patch anchor after symmetry-preserving
      grid quantization, then recolor every original dent cell to teal (8).

    Key insight:
    - The black cells must be read as shaped dent objects, not isolated pixels. The
      correct dents reveal themselves because they come in symmetric pairs inside a
      region, and each whole dent object turns into one opposite-color 3x3 patch
      while its full footprint becomes teal.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    region_ids, region_color, region_center, _region_bbox = _region_data(clean, colors)
    out = clean.copy()
    for cells_a, cells_b, region_id, host_color in _paired_dent_components(g, clean, colors):
        other = colors[1] if int(host_color) == int(colors[0]) else colors[0]
        anchor_a, anchor_b = _pair_patch_centers(cells_a, cells_b, region_center[region_id])
        _stamp_patch(out, anchor_a, int(other))
        _stamp_patch(out, anchor_b, int(other))
        for r, c in cells_a + cells_b:
            out[r, c] = Color.TEAL
    return out


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((N, N), dtype=int)
    orientation = rng.choice(["vertical", "horizontal"])

    if orientation == "vertical":
        total_h = rng.randint(max(12, int(N * 0.62)), N - 2)
        min_w = max(10, int(N * 0.24))
        max_w = max(min_w, int(N * 0.38))
        left_w = rng.randint(min_w, max_w)
        right_w = rng.randint(min_w, max_w)
        total_w = left_w + right_w
        if total_w > N - 2:
            return None
        r0 = rng.randint(1, N - total_h - 1)
        c0 = rng.randint(1, N - total_w - 1)
        r1 = r0 + total_h
        split = c0 + left_w
        c1 = split + right_w
        left_color, right_color = (color_a, color_b) if rng.random() < 0.5 else (color_b, color_a)
        grid[r0:r1, c0:split] = left_color
        grid[r0:r1, split:c1] = right_color
    else:
        total_w = rng.randint(max(12, int(N * 0.62)), N - 2)
        min_h = max(10, int(N * 0.24))
        max_h = max(min_h, int(N * 0.38))
        top_h = rng.randint(min_h, max_h)
        bottom_h = rng.randint(min_h, max_h)
        total_h = top_h + bottom_h
        if total_h > N - 2:
            return None
        r0 = rng.randint(1, N - total_h - 1)
        c0 = rng.randint(1, N - total_w - 1)
        split = r0 + top_h
        r1 = split + bottom_h
        c1 = c0 + total_w
        top_color, bottom_color = (color_a, color_b) if rng.random() < 0.5 else (color_b, color_a)
        grid[r0:split, c0:c1] = top_color
        grid[split:r1, c0:c1] = bottom_color

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / float(N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    return grid


def _within_patch_margin(center: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = center
    r0, c0, r1, c1 = bbox
    return r0 + 1 <= r <= r1 - 1 and c0 + 1 <= c <= c1 - 1


def _pair_is_spaced(
    cells_a: tuple[tuple[int, int], ...],
    cells_b: tuple[tuple[int, int], ...],
    anchors: tuple[tuple[int, int], tuple[int, int]],
    used_cells: set[tuple[int, int]],
    used_anchors: list[tuple[int, int]],
) -> bool:
    for r, c in cells_a + cells_b:
        for rr, cc in used_cells:
            if max(abs(r - rr), abs(c - cc)) < DENT_CELL_SEPARATION:
                return False
    for a in anchors:
        for b in used_anchors:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < DENT_PATCH_SEPARATION:
                return False
    if max(abs(anchors[0][0] - anchors[1][0]), abs(anchors[0][1] - anchors[1][1])) < DENT_PATCH_SEPARATION:
        return False
    return True


def _enumerate_pair_candidates(
    clean: np.ndarray,
    region_id: int,
    region_ids: np.ndarray,
    region_center: tuple[int, int],
    region_bbox: tuple[int, int, int, int],
) -> list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], tuple[int, int]]]]:
    r0, c0, r1, c1 = region_bbox
    candidates = []
    for shape in SHAPES:
        shape_h = 1 + max(dr for dr, _dc in shape)
        shape_w = 1 + max(dc for _dr, dc in shape)
        for base_r in range(r0 + 2, r1 - shape_h):
            for base_c in range(c0 + 2, c1 - shape_w):
                cells_a = tuple(sorted((base_r + dr, base_c + dc) for dr, dc in shape))
                if not all(int(region_ids[r, c]) == region_id for r, c in cells_a):
                    continue
                cells_b = _reflect_cells(cells_a, region_center)
                if cells_a >= cells_b:
                    continue
                if cells_a == cells_b:
                    continue
                if not all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells_b):
                    continue
                if not all(r0 + 2 <= r <= r1 - 2 and c0 + 2 <= c <= c1 - 2 for r, c in cells_a + cells_b):
                    continue
                if not all(int(region_ids[r, c]) == region_id for r, c in cells_b):
                    continue
                if set(cells_a) & set(cells_b):
                    continue
                ring_a = _ring_coords(cells_a, clean.shape)
                ring_b = _ring_coords(cells_b, clean.shape)
                if any(int(region_ids[r, c]) != region_id for r, c in ring_a | ring_b):
                    continue
                anchors = _pair_patch_centers(cells_a, cells_b, region_center)
                if not _within_patch_margin(anchors[0], region_bbox):
                    continue
                if not _within_patch_margin(anchors[1], region_bbox):
                    continue
                candidates.append((cells_a, cells_b, anchors))
    return candidates


def _select_pairs_for_region(
    clean: np.ndarray,
    region_id: int,
    region_ids: np.ndarray,
    region_center: tuple[int, int],
    region_bbox: tuple[int, int, int, int],
    target_pairs: int,
    rng: random.Random,
) -> list[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], tuple[int, int]]]] | None:
    all_candidates = _enumerate_pair_candidates(clean, region_id, region_ids, region_center, region_bbox)
    if not all_candidates:
        return None
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        rng.shuffle(all_candidates)
        chosen = []
        used_cells: set[tuple[int, int]] = set()
        used_anchors: list[tuple[int, int]] = []
        size_ge3 = 0
        for cells_a, cells_b, anchors in all_candidates:
            if not _pair_is_spaced(cells_a, cells_b, anchors, used_cells, used_anchors):
                continue
            chosen.append((cells_a, cells_b, anchors))
            used_cells.update(cells_a)
            used_cells.update(cells_b)
            used_anchors.extend(list(anchors))
            if len(cells_a) >= 3:
                size_ge3 += 1
            if len(chosen) == target_pairs:
                break
        if len(chosen) == target_pairs and size_ge3 >= 1:
            return chosen
    return None


def _expected_output(
    clean: np.ndarray,
    region_color: dict[int, int],
    region_center: dict[int, tuple[int, int]],
    colors: tuple[int, int],
    pairs: list[tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]],
) -> np.ndarray:
    out = clean.copy()
    for region_id, cells_a, cells_b in pairs:
        host = int(region_color[region_id])
        other = colors[1] if host == int(colors[0]) else colors[0]
        anchor_a, anchor_b = _pair_patch_centers(cells_a, cells_b, region_center[region_id])
        _stamp_patch(out, anchor_a, int(other))
        _stamp_patch(out, anchor_b, int(other))
        for r, c in cells_a + cells_b:
            out[r, c] = Color.TEAL
    return out


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    expected: np.ndarray,
    n_in_bg: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        if int(inp[r, c]) != 0:
            continue
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        trial = inp.copy()
        trial[r, c] = speck_color
        if not np.array_equal(main(trial), expected):
            continue
        inp[r, c] = speck_color
        placed += 1


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


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != host:
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


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    expected: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    h, w = clean.shape
    colors = (int(color_a), int(color_b))
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
        if int(inp[r, c]) != host:
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
        trial = inp.copy()
        trial[r, c] = other
        if int(_clean_foreground(trial, colors, Color.BLACK)[r, c]) != host:
            continue
        if not np.array_equal(main(trial), expected):
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

    colors = (int(color_a), int(color_b))
    region_ids, region_color, region_center, region_bbox = _region_data(clean, colors)
    region_list = sorted(region_color)
    if len(region_list) != 2:
        return None

    base_pairs = rng.randint(DENT_PAIRS_PER_REGION_MIN, DENT_PAIRS_PER_REGION_MAX)
    delta = rng.choice([-1, 0, 0, 1])
    target_counts = {
        region_list[0]: base_pairs,
        region_list[1]: min(DENT_PAIRS_PER_REGION_MAX, max(DENT_PAIRS_PER_REGION_MIN, base_pairs + delta)),
    }

    chosen_pairs: list[tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = []
    for region_id in region_list:
        picked = _select_pairs_for_region(
            clean,
            region_id,
            region_ids,
            region_center[region_id],
            region_bbox[region_id],
            target_counts[region_id],
            rng,
        )
        if picked is None:
            return None
        for cells_a, cells_b, _anchors in picked:
            chosen_pairs.append((region_id, cells_a, cells_b))

    expected = _expected_output(clean, region_color, region_center, colors, chosen_pairs)

    inp = clean.copy()
    for _region_id, cells_a, cells_b in chosen_pairs:
        for r, c in cells_a + cells_b:
            inp[r, c] = Color.BLACK

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, expected, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, expected, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
