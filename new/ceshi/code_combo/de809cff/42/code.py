# Combo variant of de809cff (index 42)
# combo_spec: D11/1+D18/2
# Name: 输出仅包含变换区域+重叠补丁合并

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the fused variant:
# - Concrete trace: in the base task, two nearby dents from opposite host regions
#   stamp opposite-color 3x3 patches, and the overlap color is just a write-order
#   artifact. Here the same dents first generate two 3x3 footprints, those
#   footprints are unioned into one overlap-connected patch component, the
#   component centroid is computed, and the dent center closest to that centroid
#   chooses the color for the entire merged component. Only after this merge do we
#   crop the final answer to the patch bbox expanded by one cell.
# - Complexity calibration: the generator builds one compact mixed-color cluster
#   with 5-8 dents. That keeps the crop visually small, but the cluster is large
#   enough that merge-vs-naive stamping changes multiple cells and is easy to see.
# - Hidden assumptions that no longer hold: `out = clean.copy()` is no longer the
#   final frame because unchanged bulk regions must be discarded; dent spacing can
#   no longer rely on the original separation heuristic because overlapped patches
#   are now required; overlap color cannot depend on scan order.
# - Placement legality: every selected dent stays in-bounds, every accepted set of
#   dents is verified simultaneously against the cleanup+detection logic, and the
#   chosen dents form a single overlap-connected component before noise is added.
# - Visual observability: the generator requires at least one merged component that
#   contains dents of both patch colors and differs nontrivially from naive
#   last-write stamping. The crop ratio is also kept small so the D11/1 change is
#   visible without needing many examples.


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(colors: tuple[int, int], host: int) -> int:
    return int(colors[1] if int(host) == int(colors[0]) else colors[0])


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
        other = _other_color(colors, int(host))
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _patch_cells(shape: tuple[int, int], r: int, c: int) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    cells: set[tuple[int, int]] = set()
    for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
        for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
            cells.add((int(rr), int(cc)))
    return cells


def _patches_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) <= 2 and abs(int(a[1]) - int(b[1])) <= 2


def _cluster_patch_regions(
    shape: tuple[int, int], centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> list[dict]:
    n = len(centers)
    parents = list(range(n))
    footprints = [_patch_cells(shape, r, c) for r, c, _host in centers]

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(x: int, y: int) -> None:
        rx = find(x)
        ry = find(y)
        if rx != ry:
            parents[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if footprints[i].intersection(footprints[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    components: list[dict] = []
    for indices in groups.values():
        union_cells: set[tuple[int, int]] = set()
        for idx in indices:
            union_cells |= footprints[idx]
        rows = [rr for rr, _cc in union_cells]
        cols = [cc for _rr, cc in union_cells]
        center_r = float(sum(rows)) / float(len(rows))
        center_c = float(sum(cols)) / float(len(cols))

        scored: list[tuple[float, int, int, int]] = []
        for idx in indices:
            r, c, _host = centers[idx]
            dist2 = (float(r) - center_r) ** 2 + (float(c) - center_c) ** 2
            scored.append((dist2, int(r), int(c), idx))
        scored.sort()
        winner_idx = int(scored[0][3])
        winner_host = int(centers[winner_idx][2])
        patch_colors = [_other_color(colors, int(centers[idx][2])) for idx in indices]
        components.append(
            {
                "indices": [int(idx) for idx in indices],
                "cells": union_cells,
                "winner_idx": winner_idx,
                "winner_color": _other_color(colors, winner_host),
                "patch_colors": patch_colors,
                "centroid": (center_r, center_c),
                "best_dist2": float(scored[0][0]),
                "second_dist2": float(scored[1][0]) if len(scored) > 1 else None,
            }
        )

    components.sort(key=lambda comp: (-len(comp["cells"]), min(comp["indices"])))
    return components


def _render_merged_full(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    out = clean.copy()
    patch_mask = np.zeros(clean.shape, dtype=bool)
    components = _cluster_patch_regions(clean.shape, centers, colors)

    for comp in components:
        for rr, cc in comp["cells"]:
            out[int(rr), int(cc)] = int(comp["winner_color"])
            patch_mask[int(rr), int(cc)] = True

    for r, c, _host in centers:
        out[int(r), int(c)] = int(Color.TEAL)

    return out, patch_mask, components


def _render_naive_full(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    h, w = clean.shape
    out = clean.copy()
    for r, c, host in centers:
        stamp = _other_color(colors, int(host))
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp)
    for r, c, _host in centers:
        out[int(r), int(c)] = int(Color.TEAL)
    return out


def _crop_patch_window(full: np.ndarray, patch_mask: np.ndarray, margin: int = 1) -> np.ndarray:
    coords = np.argwhere(patch_mask)
    assert coords.size > 0, "expected at least one transformed patch"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = max(0, int(r0) - int(margin))
    c0 = max(0, int(c0) - int(margin))
    r1 = min(full.shape[0] - 1, int(r1) + int(margin))
    c1 = min(full.shape[1] - 1, int(c1) + int(margin))
    return full[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _variant_diagnostics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    colors = _main_colors(g, Color.BLACK)
    clean = _clean_foreground(g, colors, Color.BLACK)
    centers = _dent_centers(g, clean, colors, Color.BLACK)
    merged_full, patch_mask, comps = _render_merged_full(clean, centers, colors)
    naive_full = _render_naive_full(clean, centers, colors)
    crop = _crop_patch_window(merged_full, patch_mask)

    merge_delta = int(np.sum((merged_full != naive_full) & patch_mask))
    mixed_clusters = sum(1 for comp in comps if len(set(comp["patch_colors"])) > 1)
    largest_cluster = max((len(comp["indices"]) for comp in comps), default=0)
    return {
        "clean": clean,
        "centers": centers,
        "merged_full": merged_full,
        "naive_full": naive_full,
        "patch_mask": patch_mask,
        "components": comps,
        "crop": crop,
        "merge_delta_cells": merge_delta,
        "mixed_clusters": int(mixed_clusters),
        "largest_cluster_dents": int(largest_cluster),
        "crop_area": int(crop.shape[0] * crop.shape[1]),
        "input_area": int(g.shape[0] * g.shape[1]),
        "crop_ratio": float(crop.shape[0] * crop.shape[1]) / float(g.shape[0] * g.shape[1]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse overlapping dent patches into merged blobs, then keep only that local window.

    Input structure:
    - Black background with two large foreground regions in two non-black colors.
    - The regions still contain black dents plus one-cell foreground noise in the
      background and across the color boundary.

    Rule:
    - First clean the foreground locally so isolated specks disappear and boundary
      contamination snaps back to the dominant neighboring region.
    - Each original black dent whose cleaned 3x3 neighborhood is dominated by one
      foreground color proposes a 3x3 patch in the opposite color.
    - Any proposed patches that overlap are merged into one connected patch region;
      the entire merged region takes the patch color of the dent center closest to
      that region's centroid, then every dent center is marked teal.
    - The final answer is cropped to the bounding box of all patch cells, expanded
      by one cell of context on each side and clipped to the grid.

    Key insight: the important object is no longer the whole cleaned canvas, but
    the compact union of transformed dent patches. You must merge overlaps before
    deciding both the patch color and the crop boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    merged_full, patch_mask, _components = _render_merged_full(clean, centers, colors)
    return _crop_patch_window(merged_full, patch_mask)


# --------------- generator (combo) ---------------

# --- CFG: canvas and region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
REGION_MARGIN_MAX = 3
MIN_REGION_CELLS = 110
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84

# --- CFG: compact overlap cluster ---
CLUSTER_DENT_COUNT_MIN, CLUSTER_DENT_COUNT_MAX = 5, 8
CLUSTER_RADIUS = 3
MIN_MERGE_DELTA = 6
MAX_CROP_RATIO = 0.28

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 6, 16
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.60, 0.82
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6
DENT_MIN_HOST_3x3 = 5

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 700


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((N, N), dtype=int)
    orientation = rng.choice(["vertical", "horizontal"])

    if orientation == "vertical":
        top_a = rng.randint(0, REGION_MARGIN_MAX)
        bot_a = rng.randint(0, REGION_MARGIN_MAX)
        top_b = rng.randint(0, REGION_MARGIN_MAX)
        bot_b = rng.randint(0, REGION_MARGIN_MAX)
        left_a = rng.randint(0, REGION_MARGIN_MAX)
        right_b = N - rng.randint(0, REGION_MARGIN_MAX)
        a_end = rng.randint(int(0.58 * N), int(0.76 * N))
        b_start = rng.randint(int(0.26 * N), int(0.44 * N))
        if b_start >= a_end - 2:
            return None
        grid[top_a : N - bot_a, left_a:a_end] = int(color_a)
        grid[top_b : N - bot_b, b_start:right_b] = int(color_b)
    else:
        left_a = rng.randint(0, REGION_MARGIN_MAX)
        right_a = rng.randint(0, REGION_MARGIN_MAX)
        left_b = rng.randint(0, REGION_MARGIN_MAX)
        right_b = rng.randint(0, REGION_MARGIN_MAX)
        top_a = rng.randint(0, REGION_MARGIN_MAX)
        bot_b = N - rng.randint(0, REGION_MARGIN_MAX)
        a_end = rng.randint(int(0.58 * N), int(0.76 * N))
        b_start = rng.randint(int(0.26 * N), int(0.44 * N))
        if b_start >= a_end - 2:
            return None
        grid[top_a:a_end, left_a : N - right_a] = int(color_a)
        grid[b_start:bot_b, left_b : N - right_b] = int(color_b)

    cnt_a = int(np.sum(grid == int(color_a)))
    cnt_b = int(np.sum(grid == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = float(cnt_a + cnt_b) / float(N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[int(r), int(c)]) != int(host):
        return False
    sub = _sub3x3(grid, int(r), int(c))
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _has_other_within(clean: np.ndarray, r: int, c: int, other: int, radius: int) -> bool:
    h, w = clean.shape
    for rr in range(max(0, int(r) - int(radius)), min(h, int(r) + int(radius) + 1)):
        for cc in range(max(0, int(c) - int(radius)), min(w, int(c) + int(radius) + 1)):
            if int(clean[int(rr), int(cc)]) == int(other):
                return True
    return False


def _all_detectable(clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]) -> bool:
    inp = clean.copy()
    for r, c, _host in centers:
        inp[int(r), int(c)] = int(Color.BLACK)
    cleaned = _clean_foreground(inp, colors, Color.BLACK)
    detected = _dent_centers(inp, cleaned, colors, Color.BLACK)
    detected_coords = {(int(r), int(c)) for r, c, _host in detected}
    wanted = {(int(r), int(c)) for r, c, _host in centers}
    return wanted == detected_coords


def _grow_cluster(
    clean: np.ndarray, colors: tuple[int, int], rng: random.Random
) -> list[tuple[int, int, int]] | None:
    color_a, color_b = map(int, colors)
    h, w = clean.shape
    valid: dict[int, list[tuple[int, int, int]]] = {color_a: [], color_b: []}

    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                if _has_other_within(clean, r, c, color_b, 2):
                    valid[color_a].append((int(r), int(c), int(color_a)))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                if _has_other_within(clean, r, c, color_a, 2):
                    valid[color_b].append((int(r), int(c), int(color_b)))

    mixed_pairs: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for a in valid[color_a]:
        for b in valid[color_b]:
            if _patches_overlap((a[0], a[1]), (b[0], b[1])):
                mixed_pairs.append((a, b))
    rng.shuffle(mixed_pairs)

    for first, second in mixed_pairs:
        anchor_r = 0.5 * float(first[0] + second[0])
        anchor_c = 0.5 * float(first[1] + second[1])
        desired = rng.randint(CLUSTER_DENT_COUNT_MIN, CLUSTER_DENT_COUNT_MAX)
        pool: list[tuple[int, int, int]] = []
        seen = {(first[0], first[1]), (second[0], second[1])}
        for cand in valid[color_a] + valid[color_b]:
            if (cand[0], cand[1]) in seen:
                continue
            if max(abs(int(cand[0]) - anchor_r), abs(int(cand[1]) - anchor_c)) > CLUSTER_RADIUS:
                continue
            pool.append(cand)
        rng.shuffle(pool)
        pool.sort(key=lambda cand: (max(abs(int(cand[0]) - anchor_r), abs(int(cand[1]) - anchor_c)), cand[0], cand[1]))

        chosen = [first, second]
        if not _all_detectable(clean, chosen, colors):
            continue

        for cand in pool:
            if len(chosen) >= desired:
                break
            if not any(_patches_overlap((cand[0], cand[1]), (r, c)) for r, c, _host in chosen):
                continue
            trial = chosen + [cand]
            if not _all_detectable(clean, trial, colors):
                continue
            chosen = trial

        if len(chosen) < CLUSTER_DENT_COUNT_MIN:
            continue

        diag_inp = clean.copy()
        for r, c, _host in chosen:
            diag_inp[int(r), int(c)] = int(Color.BLACK)
        diag = _variant_diagnostics(diag_inp)
        if len(diag["components"]) != 1:
            continue
        if diag["mixed_clusters"] < 1:
            continue
        if diag["largest_cluster_dents"] < CLUSTER_DENT_COUNT_MIN:
            continue
        if diag["merge_delta_cells"] < MIN_MERGE_DELTA:
            continue
        if diag["crop_ratio"] > MAX_CROP_RATIO:
            continue
        comp = diag["components"][0]
        if comp["second_dist2"] is not None and abs(comp["best_dist2"] - comp["second_dist2"]) < 1e-6:
            continue
        return chosen

    return None


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, colors: tuple[int, int], rng: random.Random) -> None:
    color_a, color_b = map(int, colors)
    bg_candidates = list(zip(*np.where(clean == int(Color.BLACK)), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r = int(r_)
        c = int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == int(Color.BLACK))) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != int(Color.BLACK)]
        if nearby_fg:
            speck_color = _Counter(nearby_fg).most_common(1)[0][0]
        else:
            speck_color = rng.choice([int(color_a), int(color_b)])
        inp[r, c] = int(speck_color)
        placed += 1


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[int(r) - rr, int(c) - cc] = int(color)
                if int(np.sum(block == int(color))) == 4:
                    return True
    return False


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = int(r) + dr, int(c) + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0, nc0 = max(0, nr - 1), max(0, nc - 1)
            nsub[int(r) - nr0, int(c) - nc0] = int(other)
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
            bg_n = int(np.sum(nsub == int(Color.BLACK)))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_boundary_noise(inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, colors: tuple[int, int], rng: random.Random) -> None:
    color_a, color_b = map(int, colors)
    boundary_candidates: list[tuple[int, int, int, int]] = []
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            v = int(clean[r, c])
            if v not in (color_a, color_b):
                continue
            other = color_b if v == color_a else color_a
            if int(np.sum(_sub3x3(clean, r, c) == int(other))) >= 1:
                boundary_candidates.append((int(r), int(c), int(v), int(other)))
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
        inp[r, c] = int(other)
        placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    colors = tuple(map(int, rng.sample(fg_pool, 2)))

    clean = _build_regions(N, colors[0], colors[1], rng)
    if clean is None:
        return None

    cluster = _grow_cluster(clean, colors, rng)
    if cluster is None:
        return None

    inp = clean.copy()
    for r, c, _host in cluster:
        inp[int(r), int(c)] = int(Color.BLACK)

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, colors, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, colors, rng)

    try:
        produced = main(inp)
        diag = _variant_diagnostics(inp)
    except Exception:
        return None

    if not np.array_equal(produced, diag["crop"]):
        return None
    if diag["mixed_clusters"] < 1:
        return None
    if diag["merge_delta_cells"] < MIN_MERGE_DELTA:
        return None
    if diag["crop_ratio"] > MAX_CROP_RATIO:
        return None
    if len(diag["components"]) != 1:
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
