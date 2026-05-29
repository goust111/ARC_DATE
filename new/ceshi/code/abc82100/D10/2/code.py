# Variant of abc82100
# Dimension: D10 — 源像素与规则示例的空间分布 (meta)
# Change: 规则示例和teal模板集中在网格的某个区域，源像素散布在其他区域... → 每个规则示例紧邻其对应的源像素

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the D10 variant
# -----------------------------------
# spec_variant:
#   core_change:
#     old: each teal-backed src->tgt example teaches a global color rule that is
#       applied to every scattered source pixel of the same color anywhere.
#     new: each legend applies only to one nearby source pixel. The solver must
#       pair a source pixel with the legend sitting 2-3 cells away from it.
#   concrete_pixel_layout:
#     cluster:
#       - a legend source pixel and an adjacent legend target pixel
#       - one step further in the same direction is the legend anchor, touching
#         or belonging to a teal(8) stencil component
#       - a second source pixel of the same color sits nearby at Chebyshev
#         distance 2 or 3 from the legend source, but not on the legend line
#       - output erases the legend and teal stencil, then stamps that stencil on
#         the nearby source pixel in the legend target color
#   behavior_trace:
#     old_flow:
#       - discover one legend for color A->B with stencil P
#       - find every other pixel of color A across the grid
#       - stamp P recolored to B on all of them
#     new_flow:
#       - discover one local legend for A->B with stencil P
#       - find only the A pixel that is locally attached to this legend
#       - stamp P recolored to B only there
#     consequence:
#       - repeated source colors are now safe and intentional
#       - two clusters can use the same source color but different target colors
#         or different stencils, forcing spatial pairing instead of global lookup
#   complexity_calibration:
#     - each puzzle has 2-5 local clusters, not a single trivial example
#     - at least one source color is reused across multiple clusters
#     - reused source colors always get different target colors, so the original
#       global-color solver would over-apply the legends and fail
#     - bystanders use colors outside all legend colors to avoid ambiguous cues
#   main_implicit_assumption_review:
#     - original main assumed source_color uniquely identifies one rule; this is
#       false now, so local assignment replaces global color grouping
#     - original main erased all pixels with a learned source color unless they
#       were legend pixels; this would wrongly delete unrelated local clusters,
#       so the new main erases only explicitly assigned execution sources
#     - original main could apply many legends to the same source pixel when
#       colors repeat; the new main assigns at most one execution source per
#       legend using nearest-legend disambiguation
#   placement_legality:
#     - generator places each cluster as an isolated footprint containing legend,
#       teal stencil, execution source, and final output stamp
#     - cluster footprints are kept disjoint with a one-cell moat for readability
#     - execution sources stay 2-3 cells from the legend source, and at least
#       two cells from the legend stencil/pair, so the "instruction-execution"
#       pairing is visible by eye
#     - output stamps are kept in bounds and prevented from overlapping other
#       clusters, so the scene stays interpretable
#   visual_observability:
#     - multiple clusters in one grid visibly show "same source color, different
#       nearby instruction" cases, which makes the locality rule inferable
#     - the legend and the execution source are close enough to read as one unit,
#       unlike the original separated legend area vs scattered source area
#   cleanliness:
#     - the generator avoids dense overlaps and keeps the teal stencil confined
#       to the local cluster, preventing accidental secondary legends
#   color_reasoning:
#     - no sorting or logic depends on hidden color ids
#     - color changes are always explained by an adjacent visible legend target
#   generator_guarantees_for_new_main:
#     - every intended legend has exactly one nearby execution source of the same color
#     - no bystander shares a source color
#     - for a reused source color, the nearest same-color execution source to a
#       legend is unique, so nearest-legend assignment is well-defined
#     - if a geometric coincidence creates an extra src->tgt pair, it has no
#       uniquely claimable nearby execution source and is therefore ignored


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _discover_legends(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    legends: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                legends.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    legends.sort(key=lambda item: (item["source_pos"], item["target_pos"], item["target_color"]))
    return legends


def _assign_local_sources(grid: np.ndarray, legends: list[dict], bg: int, pattern_color: int = 8) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]]]:
    legend_pixels: set[tuple[int, int]] = set()
    legends_by_color: dict[int, list[int]] = {}
    for i, legend in enumerate(legends):
        legend_pixels.add(legend["source_pos"])
        legend_pixels.add(legend["target_pos"])
        legends_by_color.setdefault(int(legend["source_color"]), []).append(i)

    positions_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in map(tuple, np.argwhere((grid != int(bg)) & (grid != int(pattern_color)))):
        pos = (int(r), int(c))
        if pos in legend_pixels:
            continue
        color = int(grid[pos])
        positions_by_color.setdefault(color, []).append(pos)

    assignments: dict[int, tuple[int, int]] = {}
    used_positions: set[tuple[int, int]] = set()
    for i, legend in enumerate(legends):
        source_color = int(legend["source_color"])
        candidates: list[tuple[int, int, int, int]] = []
        for pos in positions_by_color.get(source_color, []):
            if pos in used_positions:
                continue
            dist = _chebyshev(pos, legend["source_pos"])
            if dist not in LOCAL_SOURCE_DISTS:
                continue
            peer_dists = [_chebyshev(pos, legends[j]["source_pos"]) for j in legends_by_color[source_color]]
            if dist != min(peer_dists) or peer_dists.count(dist) != 1:
                continue
            candidates.append((dist, abs(pos[0] - legend["anchor"][0]) + abs(pos[1] - legend["anchor"][1]), pos[0], pos[1]))

        if candidates:
            candidates.sort()
            chosen = (int(candidates[0][2]), int(candidates[0][3]))
            assignments[i] = chosen
            used_positions.add(chosen)

    return assignments, legend_pixels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a nearby src->tgt+teal example to transform only its local source pixel.

    Input structure:
    - Black background with several compact clusters.
    - In each cluster, a colored source-target pair is followed by a nearby teal
      stencil, and one extra source-colored pixel sits 2-3 cells away as the
      execution site for that same local example.

    Rule:
    - For each local legend, copy its teal component onto the nearby source pixel
      of the same color, recoloring the copied component to the legend target color.
    - The legend anchor is translated onto that local source pixel; other clusters
      with the same source color do not share the rule.
    - Remove all teal components, legend pixels, and the source pixels that were
      explicitly paired with a nearby legend.

    Key insight: The puzzle no longer teaches a global color dictionary. Each
    source pixel must be matched to the legend that is physically next to it.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    legends = _discover_legends(g, bg=bg, pattern_color=pattern_color)
    assignments, legend_pixels = _assign_local_sources(g, legends, bg=bg, pattern_color=pattern_color)
    local_source_pixels = set(assignments.values())

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        pos = (int(r), int(c))
        if pos in legend_pixels or pos in local_source_pixels:
            continue
        out[pos] = int(g[pos])

    for i, legend in enumerate(legends):
        if i not in assignments:
            continue
        sr, sc = assignments[i]
        ar, ac = legend["anchor"]
        pattern = legend["pattern"]
        dr = sr - ar
        dc = sc - ac
        rows = pattern[:, 0] + dr
        cols = pattern[:, 1] + dc
        ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[ok], cols[ok]] = int(legend["target_color"])

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
MIN_DIM_BY_GROUPS = {2: 8, 3: 11, 4: 14, 5: 17}

# Local clusters
GROUP_COUNT_WEIGHTS = [2, 2, 3, 3, 4, 4, 5]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
LOCAL_SOURCE_DISTS = {2, 3}

# Stencils
STENCIL_MAX_CAP = 8
STENCIL_MIN_AREA_DIV = 120
STENCIL_MAX_AREA_DIV = 40

# Bystanders
BYSTANDER_MAX = 4

# Retry budgets
CLUSTER_RETRIES = 700
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_LOCAL_OFFSETS = [
    (-3, -1), (-3, 0), (-3, 1),
    (-2, -2), (-2, -1), (-2, 1), (-2, 2),
    (-1, -3), (-1, -2), (-1, 2), (-1, 3),
    (1, -3), (1, -2), (1, 2), (1, 3),
    (2, -2), (2, -1), (2, 1), (2, 2),
    (3, -1), (3, 0), (3, 1),
]


def _expand_with_moat(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _allocate_counts(num_groups: int, num_families: int) -> list[int]:
    counts = [1] * num_families
    for _ in range(num_groups - num_families):
        counts[random.randrange(num_families)] += 1
    random.shuffle(counts)
    return counts


def _build_groups(num_groups: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)

    max_families = min(3, num_groups - 1 if num_groups > 1 else 1)
    num_families = random.randint(1, max_families)
    src_colors = colors[:num_families]
    target_pool = [c for c in NON_BG_NON_TEAL_COLORS if c not in src_colors]
    counts = _allocate_counts(num_groups, num_families)

    groups: list[dict] = []
    used_colors: set[int] = set()
    for src, count in zip(src_colors, counts):
        targets = random.sample(target_pool, count)
        for tgt in targets:
            stencil, anchor = _make_stencil(min_s, max_s)
            rel = [(r - anchor[0], c - anchor[1]) for r, c in stencil]
            groups.append({"src": int(src), "tgt": int(tgt), "rel": rel})
            used_colors.add(int(src))
            used_colors.add(int(tgt))

    random.shuffle(groups)
    return groups, used_colors


def _offsets_for_direction(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr == 0:
        return [offset for offset in _LOCAL_OFFSETS if offset[0] != 0]
    return [offset for offset in _LOCAL_OFFSETS if offset[1] != 0]


def _min_distance_to_cells(cell: tuple[int, int], others: set[tuple[int, int]]) -> int:
    return min(_chebyshev(cell, other) for other in others)


def _place_clusters(groups: list[dict], inp: np.ndarray) -> bool:
    h, w = inp.shape
    blocked: set[tuple[int, int]] = set()
    occupied_output: set[tuple[int, int]] = set()

    for group in groups:
        rel = group["rel"]
        placed = False
        for _ in range(CLUSTER_RETRIES):
            dr, dc = random.choice(_DIRS4)
            off_r, off_c = random.choice(_offsets_for_direction(dr, dc))
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            er, ec = sr + off_r, sc + off_c

            legend_cells = {(sr, sc), (tr, tc)}
            teal_cells: set[tuple[int, int]] = set()
            stamp_cells: set[tuple[int, int]] = set()

            ok = True
            for rr, cc in rel:
                teal = (ar + rr, ac + cc)
                stamp = (er + rr, ec + cc)
                if not (0 <= teal[0] < h and 0 <= teal[1] < w and 0 <= stamp[0] < h and 0 <= stamp[1] < w):
                    ok = False
                    break
                teal_cells.add(teal)
                stamp_cells.add(stamp)
            if not ok:
                continue
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w and 0 <= er < h and 0 <= ec < w):
                continue
            instruction_cells = legend_cells | teal_cells
            if _chebyshev((er, ec), (sr, sc)) not in LOCAL_SOURCE_DISTS:
                continue
            if _min_distance_to_cells((er, ec), instruction_cells) < 2:
                continue
            if stamp_cells & instruction_cells:
                continue
            cluster_cells = instruction_cells | {(er, ec)} | stamp_cells
            if cluster_cells & blocked:
                continue
            if stamp_cells & occupied_output:
                continue

            inp[sr, sc] = group["src"]
            inp[tr, tc] = group["tgt"]
            inp[er, ec] = group["src"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8

            group["exec"] = (er, ec)
            group["stamp"] = stamp_cells
            group["footprint"] = cluster_cells
            blocked.update(_expand_with_moat(cluster_cells))
            occupied_output.update(stamp_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_bystanders(used_colors: set[int], inp: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors)))
    placed: list[tuple[int, int, int]] = []
    chosen_colors = random.sample(bystander_colors, n) if n else []
    for color in chosen_colors:
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in blocked:
                continue
            inp[r, c] = color
            placed.append((r, c, color))
            blocked.update(_expand_with_moat({(r, c)}))
            break
    return placed


def _rebuild_blocked_from_groups(groups: list[dict]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for group in groups:
        blocked.update(_expand_with_moat(set(group["footprint"])))
    return blocked


def _build_output(groups: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = color
    for group in groups:
        for r, c in group["stamp"]:
            out[r, c] = group["tgt"]
    return out


def _gen_attempt() -> np.ndarray | None:
    num_groups = random.choice(GROUP_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_GROUPS[num_groups]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    groups, used_colors = _build_groups(num_groups, h * w)
    inp = np.zeros((h, w), dtype=int)
    if not _place_clusters(groups, inp):
        return None

    blocked = _rebuild_blocked_from_groups(groups)
    bystanders = _place_bystanders(used_colors, inp, blocked)
    output = _build_output(groups, bystanders, h, w)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
