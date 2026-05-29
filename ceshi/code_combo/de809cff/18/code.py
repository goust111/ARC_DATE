# Combo variant of de809cff (index 18)
# combo_spec: D2/1+D16/0
# Name: 位置编码中心色+凹痕对称分布

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np
from scipy.ndimage import distance_transform_cdt, label

from arc_utils import Color

# Design analysis for the fused variant:
# 1. Concrete behavior trace:
#    In the base task, every detected black dent becomes an opposite-color 3x3 patch
#    and its center is always teal. Here the same cleanup and dent detection still
#    happens first, but each connected host region then chooses its own central axis
#    (horizontal or vertical through the region bbox center). Dent centers are repaired
#    into symmetric pairs on that axis: a lone detected dent can create its missing
#    mirror patch, while a dent without a compatible mirror position is discarded.
#    Finally, the center color is position-coded: cells within two taxicab layers of
#    the host-region edge are teal; deeper cells are maroon.
# 2. Complexity calibration:
#    The center-color rule is intentionally binary (teal vs maroon) rather than using
#    many depth bands. The symmetry axis is also limited to the horizontal/vertical
#    center lines of one connected region. This keeps the rule visually learnable while
#    still making the fused samples meaningfully different from the original task.
# 3. Hidden assumptions broken by the combo:
#    The original solver assumed every valid dent is kept and all centers share one
#    marker color. That no longer holds. After cleanup, dents must be assigned to a
#    specific connected host component, mirrored on that component's axis, and colored
#    from that component's edge-depth map. The generator therefore guarantees:
#    - every active component has a clear preferred axis,
#    - mirrored dent positions are also valid dent cells,
#    - paired dents share the same depth class,
#    - chosen output dent centers stay >=3 cells apart so 3x3 patches remain legible.
# 4. Placement legality and neatness:
#    Generation is still output-first. Dents are chosen only from safe host cells
#    recognized by the solver after cleanup. Symmetric orbit selection checks bounds,
#    non-overlap, and class compatibility, and noise is added only after the intended
#    dent set has been fixed.
# 5. Human-observable signal:
#    Each sample contains repeated symmetric dent pairs plus at least one repaired
#    half-pair and both center-color classes. That creates visible evidence for both
#    subrules: symmetry completion/removal and near-edge vs interior center coloring.


EDGE_TEAL_MAX = 2


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(colors: tuple[int, int], color: int) -> int:
    return int(colors[1]) if int(color) == int(colors[0]) else int(colors[0])


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
        other = _other_color(colors, int(host))
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= 5 and host_count > int(np.sum(sub == int(other)))


def _assign_component(labels: np.ndarray, r: int, c: int) -> int:
    h, w = labels.shape
    sub = labels[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
    vals = [int(v) for v in sub.flat if int(v) != 0]
    if not vals:
        return 0
    return int(_Counter(vals).most_common(1)[0][0])


def _mirror(axis_kind: str, axis_k: int, pos: tuple[int, int]) -> tuple[int, int]:
    r, c = map(int, pos)
    if axis_kind == "vertical":
        return r, int(axis_k) - c
    return int(axis_k) - r, c


def _sorted_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _component_infos(clean: np.ndarray, colors: tuple[int, int]) -> dict[tuple[int, int], dict]:
    infos: dict[tuple[int, int], dict] = {}
    for color in colors:
        host = int(color)
        other = _other_color(colors, host)
        mask = clean == host
        labels, n_labels = label(mask)
        depth = distance_transform_cdt(mask, metric="taxicab")
        for comp_id in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == comp_id)
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            valid_centers: dict[tuple[int, int], int] = {}
            for rr, cc in coords:
                r = int(rr)
                c = int(cc)
                if _is_safe_dent_center(clean, r, c, host, other):
                    valid_centers[(r, c)] = Color.TEAL if int(depth[r, c]) <= EDGE_TEAL_MAX else Color.MAROON
            infos[(host, comp_id)] = {
                "host": host,
                "other": other,
                "labels": labels,
                "comp_id": int(comp_id),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "valid_colors": valid_centers,
            }
    return infos


def _component_axes(info: dict) -> list[tuple[str, int]]:
    r0, c0, r1, c1 = info["bbox"]
    return [("vertical", int(c0 + c1)), ("horizontal", int(r0 + r1))]


def _axis_score(
    info: dict, present: set[tuple[int, int]], axis_kind: str, axis_k: int
) -> tuple[int, int, int]:
    valid = info["valid_colors"]
    score = 0
    complete_orbits = 0
    compatible_orbits = 0
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for p in present:
        if p not in valid:
            score -= 2
            continue
        q = _mirror(axis_kind, axis_k, p)
        orbit = _sorted_pair(p, q)
        if orbit in seen:
            continue
        seen.add(orbit)
        if q == p:
            score -= 3
            continue
        if q not in valid or int(valid[q]) != int(valid[p]):
            score -= 2
            continue
        compatible_orbits += 1
        score += 4
        if q in present:
            complete_orbits += 1
            score += 3
        else:
            score += 1
    return score, compatible_orbits, complete_orbits


def _best_axis_for_component(info: dict, present: set[tuple[int, int]]) -> tuple[str, int]:
    best_axis = _component_axes(info)[0]
    best_key: tuple[int, int, int] | None = None
    for axis_kind, axis_k in _component_axes(info):
        score_key = _axis_score(info, present, axis_kind, axis_k)
        if best_key is None or score_key > best_key:
            best_key = score_key
            best_axis = (axis_kind, axis_k)
    return best_axis


def _repair_component_centers(
    info: dict, present: set[tuple[int, int]], axis_kind: str, axis_k: int
) -> list[tuple[int, int, int]]:
    valid = info["valid_colors"]
    repaired: set[tuple[int, int]] = set()
    for p in sorted(present):
        if p not in valid:
            continue
        q = _mirror(axis_kind, axis_k, p)
        if q == p:
            continue
        if q not in valid or int(valid[q]) != int(valid[p]):
            continue
        repaired.add(p)
        repaired.add(q)
    return [(r, c, int(valid[(r, c)])) for r, c in sorted(repaired)]


def _repaired_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int, int]]:
    detected = _dent_centers(original, clean, colors, bg)
    restored = _restore_detected_dents(clean, detected)
    return _repaired_centers_from_detected(detected, restored, colors)


def _restore_detected_dents(
    clean: np.ndarray, detected: list[tuple[int, int, int]]
) -> np.ndarray:
    restored = clean.copy()
    for r, c, host in detected:
        restored[int(r), int(c)] = int(host)
    return restored


def _group_detected_by_component(
    detected: list[tuple[int, int, int]],
    restored: np.ndarray,
    colors: tuple[int, int],
    infos: dict[tuple[int, int], dict] | None = None,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    infos = _component_infos(restored, colors) if infos is None else infos
    labels_by_host = {int(color): info["labels"] for (color, _), info in infos.items()}
    grouped: dict[tuple[int, int], set[tuple[int, int]]] = {key: set() for key in infos}
    for r, c, host in detected:
        comp_id = _assign_component(labels_by_host[int(host)], int(r), int(c))
        if comp_id != 0 and (int(host), int(comp_id)) in grouped:
            grouped[(int(host), int(comp_id))].add((int(r), int(c)))
    return grouped


def _repaired_centers_from_detected(
    detected: list[tuple[int, int, int]], restored: np.ndarray, colors: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    infos = _component_infos(restored, colors)
    grouped = _group_detected_by_component(detected, restored, colors, infos)

    repaired: list[tuple[int, int, int, int]] = []
    for key, present in grouped.items():
        if not present:
            continue
        info = infos[key]
        axis_kind, axis_k = _best_axis_for_component(info, present)
        for r, c, center_color in _repair_component_centers(info, present, axis_kind, axis_k):
            repaired.append((int(r), int(c), int(info["host"]), int(center_color)))
    return repaired


def _stamp_output(
    clean: np.ndarray, colors: tuple[int, int], centers: list[tuple[int, int, int, int]]
) -> np.ndarray:
    out = clean.copy()
    h, w = clean.shape
    for r, c, host, _center_color in centers:
        other = _other_color(colors, host)
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
    for r, c, _host, center_color in centers:
        out[int(r), int(c)] = int(center_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn cleaned black dents into symmetric opposite-color patches with depth-coded centers.

    Input structure:
    - Black background with two large foreground colors.
    - The colored regions still contain the original one-cell noise and black dents,
      but the dents are arranged so that each host component suggests a horizontal
      or vertical symmetry axis through its center.

    Rule:
    - First denoise the foreground exactly as in the base task: isolated specks fall
      back to black and one-cell cross-color protrusions are restored to their host.
    - Detect black dents by looking for original black cells whose cleaned 3x3
      neighborhood is dominated by one foreground color.
    - Within each cleaned connected foreground component, choose the better central
      axis (horizontal or vertical), repair the dent set into symmetric pairs on that
      axis, and stamp an opposite-color 3x3 patch for every repaired dent center.
    - Mark each repaired center by edge depth: teal (8) if it lies within two taxicab
      layers of the host-region edge, maroon (9) if it lies deeper inside.

    Key insight: The output is not just “stamp every black hole”. One must first
    recover the clean host regions, then infer each component's dent symmetry, and
    only then read the center color from how close that dent sits to the region edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    base_clean = _clean_foreground(g, colors, bg)
    detected = _dent_centers(g, base_clean, colors, bg)
    restored = _restore_detected_dents(base_clean, detected)
    centers = _repaired_centers_from_detected(detected, restored, colors)
    return _stamp_output(restored, colors, centers)


# --------------- generator (combo) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
OUTPUT_ORBIT_COUNT_MIN, OUTPUT_ORBIT_COUNT_MAX = 7, 10
DENT_PATCH_SEPARATION = 3
MIN_ORBITS_PER_ACTIVE_COMPONENT = 2
HALF_PAIR_PROB = 0.45
ROGUE_SINGLE_PROB = 0.0
MANIFEST_ATTEMPTS = 16

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 500


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
        grid[r0:r1, c0:c1] = int(color)

    cnt_a = int(np.sum(grid == int(color_a)))
    cnt_b = int(np.sum(grid == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / float(N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3:
        _, n_components = label(grid == int(rect_colors[0]))
        if int(n_components) < 2:
            return None
    return grid


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = int(color)
                if int(np.sum(block == int(color))) == 4:
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
            nsub[r - nr0, c - nc0] = int(other)
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, colors: tuple[int, int], rng: random.Random) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)
    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r = int(r_)
        c = int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice(list(colors))
        inp[r, c] = int(speck_color)
        placed += 1


def _add_boundary_noise(
    inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, colors: tuple[int, int], rng: random.Random
) -> None:
    color_a, color_b = map(int, colors)
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
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
        inp[r, c] = int(other)
        placed += 1


def _orbit_candidates_for_axis(info: dict, axis_kind: str, axis_k: int) -> list[dict]:
    valid = info["valid_colors"]
    orbits: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for p, center_color in sorted(valid.items()):
        q = _mirror(axis_kind, axis_k, p)
        if q == p or q not in valid or int(valid[q]) != int(center_color):
            continue
        if _cheb(p, q) < DENT_PATCH_SEPARATION:
            continue
        orbit = _sorted_pair(p, q)
        if orbit in seen:
            continue
        seen.add(orbit)
        orbits.append(
            {
                "cells": orbit,
                "center_color": int(center_color),
                "host": int(info["host"]),
                "axis_kind": axis_kind,
                "axis_k": int(axis_k),
                "component_key": (int(info["host"]), int(info["comp_id"])),
            }
        )
    return orbits


def _axis_options(info: dict) -> list[dict]:
    options: list[dict] = []
    for axis_kind, axis_k in _component_axes(info):
        orbits = _orbit_candidates_for_axis(info, axis_kind, axis_k)
        if len(orbits) >= MIN_ORBITS_PER_ACTIVE_COMPONENT:
            options.append({"axis_kind": axis_kind, "axis_k": int(axis_k), "orbits": orbits})
    return options


def _far_enough(cells: tuple[tuple[int, int], ...], used: list[tuple[int, int]]) -> bool:
    return all(_cheb(cell, other) >= DENT_PATCH_SEPARATION for cell in cells for other in used)


def _pick_spaced_orbits(orbits: list[dict], count: int, used: list[tuple[int, int]], rng: random.Random) -> list[dict]:
    pool = list(orbits)
    rng.shuffle(pool)
    chosen: list[dict] = []
    local_used = list(used)
    for orbit in pool:
        cells = tuple(orbit["cells"])
        if not _far_enough(cells, local_used):
            continue
        chosen.append(orbit)
        local_used.extend(cells)
        if len(chosen) >= count:
            break
    return chosen


def _chosen_center_colors(orbits: list[dict]) -> list[int]:
    return [int(orbit["center_color"]) for orbit in orbits]


def _manifest_input_dents(
    selected_orbits: list[dict], chosen_axes: dict[tuple[int, int], tuple[str, int]], infos: dict[tuple[int, int], dict], rng: random.Random
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    input_dents: list[tuple[int, int]] = []
    output_centers: list[tuple[int, int]] = []
    half_pairs = 0
    for orbit in selected_orbits:
        a, b = orbit["cells"]
        output_centers.extend([a, b])
        if rng.random() < HALF_PAIR_PROB:
            input_dents.append(rng.choice([a, b]))
            half_pairs += 1
        else:
            input_dents.extend([a, b])

    rogue: list[tuple[int, int]] = []
    if rng.random() < ROGUE_SINGLE_PROB:
        rogue_pool: list[tuple[int, int]] = []
        used = list(output_centers)
        used.extend(input_dents)
        for key, info in infos.items():
            axis_kind, axis_k = chosen_axes[key]
            valid = info["valid_colors"]
            for p, center_color in valid.items():
                q = _mirror(axis_kind, axis_k, p)
                if q in valid and int(valid[q]) == int(center_color) and q != p:
                    continue
                if _far_enough((p,), used):
                    rogue_pool.append(p)
        if rogue_pool:
            rogue = [rng.choice(rogue_pool)]
            input_dents.extend(rogue)
    return input_dents, output_centers, half_pairs


def _build_target_output(clean: np.ndarray, colors: tuple[int, int], selected_orbits: list[dict]) -> np.ndarray:
    centers: list[tuple[int, int, int, int]] = []
    for orbit in selected_orbits:
        for r, c in orbit["cells"]:
            centers.append((int(r), int(c), int(orbit["host"]), int(orbit["center_color"])))
    return _stamp_output(clean, colors, centers)


def _pick_component_plan(clean: np.ndarray, colors: tuple[int, int], rng: random.Random) -> tuple[dict, list[dict]] | None:
    infos = _component_infos(clean, colors)
    axes_by_component: dict[tuple[int, int], dict] = {}
    viable_keys: list[tuple[int, int]] = []
    for key, info in infos.items():
        options = _axis_options(info)
        if not options:
            continue
        rng.shuffle(options)
        axes_by_component[key] = max(options, key=lambda opt: len(opt["orbits"]))
        viable_keys.append(key)

    colors_present = {int(key[0]) for key in viable_keys}
    if colors_present != {int(colors[0]), int(colors[1])}:
        return None

    active_keys = list(viable_keys)
    rng.shuffle(active_keys)
    selected_orbits: list[dict] = []
    used_cells: list[tuple[int, int]] = []
    target_orbits = rng.randint(OUTPUT_ORBIT_COUNT_MIN, OUTPUT_ORBIT_COUNT_MAX)

    for key in active_keys:
        base_orbits = _pick_spaced_orbits(axes_by_component[key]["orbits"], MIN_ORBITS_PER_ACTIVE_COMPONENT, used_cells, rng)
        if len(base_orbits) < MIN_ORBITS_PER_ACTIVE_COMPONENT:
            return None
        selected_orbits.extend(base_orbits)
        for orbit in base_orbits:
            used_cells.extend(list(orbit["cells"]))

    extras_pool: list[dict] = []
    for key in active_keys:
        chosen_set = {tuple(orbit["cells"]) for orbit in selected_orbits if orbit["component_key"] == key}
        for orbit in axes_by_component[key]["orbits"]:
            if tuple(orbit["cells"]) not in chosen_set:
                extras_pool.append(orbit)
    extras = _pick_spaced_orbits(extras_pool, max(0, target_orbits - len(selected_orbits)), used_cells, rng)
    selected_orbits.extend(extras)

    center_colors = _chosen_center_colors(selected_orbits)
    if Color.TEAL not in center_colors or Color.MAROON not in center_colors:
        return None

    return (
        {key: (axes_by_component[key]["axis_kind"], axes_by_component[key]["axis_k"]) for key in active_keys},
        selected_orbits,
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c not in (Color.TEAL, Color.MAROON)]
    colors = tuple(map(int, rng.sample(fg_pool, 2)))
    clean = _build_regions(N, colors[0], colors[1], rng)
    if clean is None:
        return None

    picked = _pick_component_plan(clean, colors, rng)
    if picked is None:
        return None
    chosen_axes, selected_orbits = picked
    infos = _component_infos(clean, colors)

    input_dents: list[tuple[int, int]] | None = None
    for _ in range(MANIFEST_ATTEMPTS):
        candidate_dents, _output_centers, half_pairs = _manifest_input_dents(selected_orbits, chosen_axes, infos, rng)
        if half_pairs == 0:
            continue
        probe = clean.copy()
        for r, c in candidate_dents:
            probe[int(r), int(c)] = Color.BLACK
        probe_detected = _dent_centers(probe, clean, colors, Color.BLACK)
        probe_restored = _restore_detected_dents(clean, probe_detected)
        probe_infos = _component_infos(probe_restored, colors)
        probe_present = _group_detected_by_component(probe_detected, probe_restored, colors, probe_infos)
        axis_ok = True
        for key, axis in chosen_axes.items():
            present = probe_present[key]
            if not present:
                axis_ok = False
                break
            if _best_axis_for_component(probe_infos[key], present) != axis:
                axis_ok = False
                break
        if not axis_ok:
            continue
        probe_repaired = _repaired_centers_from_detected(probe_detected, probe_restored, colors)
        probe_present_set = {pos for group in probe_present.values() for pos in group}
        probe_repaired_set = {(int(r), int(c)) for r, c, _host, _center in probe_repaired}
        if not (probe_repaired_set - probe_present_set):
            continue
        input_dents = candidate_dents
        break
    if input_dents is None:
        return None

    inp = clean.copy()
    for r, c in input_dents:
        inp[int(r), int(c)] = Color.BLACK

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, colors, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, colors, rng)

    try:
        produced = main(inp)
    except Exception:
        return None

    solved_clean = _clean_foreground(inp, colors, Color.BLACK)
    detected = _dent_centers(inp, solved_clean, colors, Color.BLACK)
    solved_restored = _restore_detected_dents(solved_clean, detected)
    repaired = _repaired_centers_from_detected(detected, solved_restored, colors)
    expected = _stamp_output(solved_restored, colors, repaired)
    if not np.array_equal(produced, expected):
        return None

    solved_infos = _component_infos(solved_restored, colors)
    present_grouped = _group_detected_by_component(detected, solved_restored, colors, solved_infos)
    present_set = {pos for group in present_grouped.values() for pos in group}
    produced_set = {(int(r), int(c)) for r, c, _host, _center in repaired}
    if not (produced_set - present_set):
        return None
    if not any(int(produced[r, c]) == Color.MAROON for r, c in produced_set):
        return None
    if not any(int(produced[r, c]) == Color.TEAL for r, c in produced_set):
        return None

    if any(_cheb(a, b) < DENT_PATCH_SEPARATION for i, a in enumerate(sorted(produced_set)) for b in sorted(produced_set)[i + 1 :]):
        return None

    for key, present in present_grouped.items():
        if not present:
            continue
        axes = _component_axes(solved_infos[key])
        scores = [_axis_score(solved_infos[key], present, axis_kind, axis_k) for axis_kind, axis_k in axes]
        ordered = sorted(scores, reverse=True)
        if len(ordered) >= 2 and ordered[0][0] - ordered[1][0] < 2:
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
