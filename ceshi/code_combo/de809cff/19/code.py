# Combo variant of de809cff (index 19)
# combo_spec: D10/2+D12/0
# Name: 多像素凹痕连通区域+补丁大小由凹痕总数决定

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


# Design analysis for the fused variant:
# - Original behavior treated every black pixel as an independent dent and always
#   stamped a 3x3 opposite-color patch around that pixel.
# - Here a dent is a 2-4 cell 4-connected black component. The solver must first
#   recover components, then assign each component to its host color region from
#   the cleaned neighborhood ring, and only then count dents per region.
# - The patch size depends on the per-region dent count, so local stamping now
#   depends on a global statistic. To keep the rule visually inferable, the
#   generator forces the two regions to have clearly different dent counts:
#   fewer-dent region -> 5x5 patches, more-dent region -> 1x1 patches, tie -> 3x3.
# - Several original assumptions no longer hold: "one black pixel == one dent",
#   "dent center is the black cell itself", and "patch size is fixed". The new
#   solver explicitly replaces each of those assumptions.
# - Generator legality is tighter than the original task. Each dent component is
#   placed fully inside a 5x5 host-colored interior zone so it stays disconnected
#   from the outer black background, remains visually recognizable, and survives
#   the cleanup stage without ambiguity. Centers are spaced so dent components do
#   not merge and large 5x5 patches stay readable.


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


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
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


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _black_components(g: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(bg):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == int(bg):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _representative_cell(cells: list[tuple[int, int]]) -> tuple[int, int]:
    mean_r = float(np.mean([r for r, _ in cells]))
    mean_c = float(np.mean([c for _, c in cells]))
    target = (_round_half_up(mean_r), _round_half_up(mean_c))
    return min(cells, key=lambda rc: (abs(rc[0] - target[0]) + abs(rc[1] - target[1]), rc[0], rc[1]))


def _detect_dents(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[dict[str, object]]:
    h, w = original.shape
    dents: list[dict[str, object]] = []
    for comp in _black_components(original, bg):
        if not (2 <= len(comp) <= 4):
            continue
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue

        comp_set = set(comp)
        counts = {int(color): 0 for color in colors}
        bg_ring = 0
        for r, c in comp:
            for nr, nc in _neighbors8(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in comp_set:
                    continue
                v = int(clean[nr, nc])
                if v in counts:
                    counts[v] += 1
                elif v == int(bg):
                    bg_ring += 1

        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] < 8:
            continue
        if counts[int(host)] <= counts[int(other)] + bg_ring:
            continue

        center_r, center_c = _representative_cell(comp)
        dents.append(
            {
                "cells": comp,
                "center": (int(center_r), int(center_c)),
                "host": int(host),
            }
        )
    return dents


def _patch_sizes(colors: tuple[int, int], dents: list[dict[str, object]]) -> dict[int, int]:
    counts = {int(color): 0 for color in colors}
    for dent in dents:
        counts[int(dent["host"])] += 1
    a, b = map(int, colors)
    if counts[a] == counts[b]:
        return {a: 3, b: 3}
    if counts[a] < counts[b]:
        return {a: 5, b: 1}
    return {a: 1, b: 5}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn multi-cell black dents into opposite-color markers whose size depends on
    how many dents the host region contains.

    Input structure:
    - Black background with two large foreground-color regions.
    - Each region contains several small black dents, but every dent is now a
      2-4 cell connected component rather than a single black pixel.
    - The grid also contains one-cell foreground noise in the background or
      across the region boundary, just as in the original task.

    Rule:
    - First clean the two foreground regions locally: isolated specks go back to
      black, and one-cell cross-boundary contamination reverts to its host color.
    - Then detect each black 2-4 cell dent component in the original grid,
      infer which foreground region surrounds it, and count dents per region.
    - Dents in the region with fewer dents get a 5x5 opposite-color patch;
      dents in the region with more dents get only a 1x1 patch; if both regions
      have the same dent count, both use 3x3. The chosen patch is centered on a
      representative cell near the dent component's centroid, and that center is
      recolored teal (8).

    Key insight: A black hole is no longer processed cell-by-cell. You must group
    black cells into dent components first, use those grouped dents to compare the
    two regions' total dent counts, and only then know how large each patch is.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    dents = _detect_dents(g, clean, colors, bg)
    sizes = _patch_sizes(colors, dents)

    out = clean.copy()
    for dent in dents:
        r, c = dent["center"]
        host = int(dent["host"])
        other = colors[1] if host == int(colors[0]) else colors[0]
        size = int(sizes[host])
        rad = size // 2
        out[max(0, r - rad) : min(g.shape[0], r + rad + 1), max(0, c - rad) : min(g.shape[1], c + rad + 1)] = int(other)

    for dent in dents:
        r, c = dent["center"]
        out[int(r), int(c)] = Color.TEAL

    return out


# --------------- generator ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 22, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.48, 0.78
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 120
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84

# --- CFG: dent placement ---
LOW_DENT_RANGE = (3, 5)
HIGH_DENT_RANGE = (8, 10)
LOW_CENTER_SEP = 6
HIGH_CENTER_SEP = 4
MIN_CELL_GAP = 1

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 20
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, -1)),
    ((0, 0), (-1, 0)),
    ((0, 0), (0, -1), (-1, 0)),
    ((0, 0), (0, -1), (0, 1)),
    ((0, 0), (-1, 0), (1, 0)),
    ((0, 0), (0, -1), (-1, 0), (-1, -1)),
    ((0, 0), (0, -1), (0, 1), (-1, 0)),
    ((0, 0), (0, -1), (0, 1), (0, -2)),
    ((0, 0), (-1, 0), (1, 0), (-2, 0)),
)


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
        rect_colors = [first, color_b if first == color_a else color_a, first]

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
    return grid


def _window_all_color(grid: np.ndarray, r: int, c: int, radius: int, color: int) -> bool:
    h, w = grid.shape
    if r - radius < 0 or r + radius >= h or c - radius < 0 or c + radius >= w:
        return False
    return bool(np.all(grid[r - radius : r + radius + 1, c - radius : c + radius + 1] == int(color)))


def _shape_cells(center: tuple[int, int], shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in shape]


def _cells_far_enough(new_cells: list[tuple[int, int]], existing: list[tuple[int, int]], gap: int) -> bool:
    for r, c in new_cells:
        for rr, cc in existing:
            if max(abs(r - rr), abs(c - cc)) <= gap:
                return False
    return True


def _place_dents_for_color(
    clean: np.ndarray, host: int, n_dents: int, center_sep: int, rng: random.Random
) -> list[dict[str, object]] | None:
    candidates = [(r, c) for r, c in zip(*np.where(clean == int(host)), strict=False) if _window_all_color(clean, int(r), int(c), 2, host)]
    candidates = [(int(r), int(c)) for r, c in candidates]
    rng.shuffle(candidates)

    chosen: list[dict[str, object]] = []
    centers: list[tuple[int, int]] = []
    occupied: list[tuple[int, int]] = []
    for center in candidates:
        if any(max(abs(center[0] - rr), abs(center[1] - cc)) < center_sep for rr, cc in centers):
            continue
        shapes = list(SHAPES)
        rng.shuffle(shapes)
        for shape in shapes:
            cells = _shape_cells(center, shape)
            if not all(0 <= r < clean.shape[0] and 0 <= c < clean.shape[1] for r, c in cells):
                continue
            if not all(0 < r < clean.shape[0] - 1 and 0 < c < clean.shape[1] - 1 for r, c in cells):
                continue
            if not all(int(clean[r, c]) == int(host) for r, c in cells):
                continue
            if not _cells_far_enough(cells, occupied, MIN_CELL_GAP):
                continue
            chosen.append({"cells": cells, "center": center, "host": int(host)})
            centers.append(center)
            occupied.extend(cells)
            break
        if len(chosen) >= n_dents:
            return chosen
    return None


def _stamp_output(clean: np.ndarray, colors: tuple[int, int], dents: list[dict[str, object]]) -> np.ndarray:
    out = clean.copy()
    sizes = _patch_sizes(colors, dents)
    for dent in dents:
        r, c = dent["center"]
        host = int(dent["host"])
        other = colors[1] if host == int(colors[0]) else colors[0]
        size = int(sizes[host])
        rad = size // 2
        out[max(0, r - rad) : min(out.shape[0], r + rad + 1), max(0, c - rad) : min(out.shape[1], c + rad + 1)] = int(other)
    for dent in dents:
        r, c = dent["center"]
        out[int(r), int(c)] = Color.TEAL
    return out


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    bg_candidates = [(int(r), int(c)) for r, c in bg_candidates]
    rng.shuffle(bg_candidates)

    placed = 0
    for r, c in bg_candidates:
        if placed >= n_in_bg:
            break
        if int(np.sum(_sub3x3(inp, r, c) == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
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
    boundary_candidates = []
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
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


def _choose_region_counts(rng: random.Random) -> tuple[int, int]:
    low = rng.randint(LOW_DENT_RANGE[0], LOW_DENT_RANGE[1])
    high = rng.randint(HIGH_DENT_RANGE[0], HIGH_DENT_RANGE[1])
    return low, high


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    low_count, high_count = _choose_region_counts(rng)
    if rng.random() < 0.5:
        target_counts = {color_a: low_count, color_b: high_count}
    else:
        target_counts = {color_a: high_count, color_b: low_count}

    dents_a = _place_dents_for_color(
        clean, color_a, target_counts[color_a], LOW_CENTER_SEP if target_counts[color_a] == low_count else HIGH_CENTER_SEP, rng
    )
    dents_b = _place_dents_for_color(
        clean, color_b, target_counts[color_b], LOW_CENTER_SEP if target_counts[color_b] == low_count else HIGH_CENTER_SEP, rng
    )
    if dents_a is None or dents_b is None:
        return None

    dents = dents_a + dents_b

    inp = clean.copy()
    for dent in dents:
        for r, c in dent["cells"]:
            inp[int(r), int(c)] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        clean_after = _clean_foreground(inp, (color_a, color_b), 0)
        detected = _detect_dents(inp, clean_after, (color_a, color_b), 0)
        produced = main(inp)
    except Exception:
        return None

    detected_counts = {color_a: 0, color_b: 0}
    for dent in detected:
        detected_counts[int(dent["host"])] += 1
    if detected_counts != target_counts:
        return None

    expected = _stamp_output(clean_after, (color_a, color_b), detected)
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
