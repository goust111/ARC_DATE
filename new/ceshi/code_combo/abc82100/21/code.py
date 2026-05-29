# Combo variant of abc82100 (index 21)
# combo_spec: D10/2+D14/1
# Name: 局部规则-源配对+特殊标记锚点

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# combo_variant:
#   summary:
#     - Keep the base task's visible "src -> tgt + nearby template" teaching style.
#     - Replace global color lookup with local instruction-execution pairing.
#     - Replace the implicit anchor with an explicit marker pixel inside each template.
#   concrete_pixel_layout:
#     cluster:
#       - legend source S and legend target T are 4-neighbors.
#       - one step farther in the same direction is a probe cell P that must be teal.
#       - the touched template is an 8-connected mixed component: teal(8) cells plus
#         exactly one blue(1) marker cell M.
#       - M is the true anchor; P is only a teal entry point into that component.
#       - one extra source-colored execution pixel E sits 2-3 king steps from S,
#         outside the legend line, so the whole cluster reads as one local package.
#   behavior_trace:
#     example:
#       - legend pair: S=(5,5), T=(5,6), probe P=(5,7)
#       - template cells: teal at (5,7) and (4,8), marker M at (4,7)
#       - execution source: E=(3,4) with the same color as S
#       - base task would stamp on every same-colored source globally and would align
#         the continued pair direction cell (5,7) to each source.
#       - D10-only logic would correctly keep the action local, but would still align
#         the template as if anchor=(5,7), shifting the output one row too low.
#       - D14-only logic would read anchor=M=(4,7) correctly, but would still stamp
#         every same-colored source in the grid.
#       - combo logic must do both changes together: pair E only with this nearby
#         legend, then translate the teal cells so M lands on E.
#   complexity_calibration:
#     - each grid has 2-5 local clusters so the locality rule is visible repeatedly.
#     - at least one source color is reused by multiple clusters, and reused clusters
#       get different targets, so the original global-color solver would fail.
#     - every template contains a marker that is not the probe cell, so the explicit
#       anchor meaning is also necessary and visually testable.
#     - the marker color is dedicated; source/target colors come from a small palette.
#   main_implicit_assumption_review:
#     - old discovery assumed "anchor = target continuation cell"; now discovery must
#       first locate the mixed template component from the teal probe, then read the
#       component's unique marker as the anchor.
#     - old application assumed one rule per source color; now repeated source colors
#       are intentional, so rules must be matched by local proximity instead.
#     - old cleanup erased every learned source color; now only explicitly assigned
#       execution pixels should disappear, otherwise another local cluster could be lost.
#     - old pattern extraction used pure teal components; now extraction must treat
#       teal+marker as one component but stamp only the teal subset.
#   placement_legality:
#     - each mixed template component has exactly one marker and at least one teal cell.
#     - the continuation probe is always a teal cell, never the marker itself.
#     - cluster footprints include legend, template, execution source, and intended
#       output stamp; footprints stay disjoint with a one-cell moat for readability.
#     - execution sources are placed at Chebyshev distance 2 or 3 from their legend
#       source and at least two cells away from the instruction cells.
#   visual_observability:
#     - repeated source colors paired with different nearby targets make the local
#       pairing rule inferable from a few examples.
#     - the blue marker appears inside every template, so humans can see that the
#       copied shape aligns by that marker rather than by the probe cell.
#   cleanliness_and_inference:
#     - bystanders never use source or target colors from any cluster, preventing
#       accidental extra pairings or ambiguous color stories.
#     - output stamps are kept in-bounds and non-overlapping across clusters so the
#       final transformed shapes remain clean and separable.
# assumptions_enforced_by_generator:
#   - every intended legend has exactly one uniquely nearest same-colored execution pixel.
#   - no bystander shares any source color.
#   - no two mixed templates touch, even diagonally.
#   - every repeated source color appears in at least two distinct local clusters.
# new_main_assumption_audit:
#   - _discover_legends assumes the continuation probe is teal, not merely adjacent to teal;
#     generator guarantees this by solving template placement from a chosen teal touch cell.
#   - _discover_pattern_components assumes each mixed component has one marker exactly;
#     generator never places marker-colored pixels outside templates.
#   - _assign_local_sources assumes each real legend has one uniquely nearest same-colored
#     non-legend pixel at Chebyshev distance 2 or 3; blocked cluster placement enforces that.
#   - the preservation loop assumes any remaining non-template colored pixels are bystanders;
#     generator enforces bystander colors outside the full source-color set.
#   - the stamping loop assumes applying a legend to its assigned source is independent of
#     processing order; generator keeps output stamps disjoint, though the solver itself would
#     still be safe under overlap because later writes are same-color within a cluster.


MARKER_COLOR = 1
PATTERN_COLOR = 8

GRID_DIM_MAX = 20
MIN_DIM_BY_GROUPS = {2: 8, 3: 11, 4: 14, 5: 17}

GROUP_COUNT_WEIGHTS = [2, 2, 3, 3, 4, 4, 5]
PAIR_COLORS = [2, 3, 4, 5, 6, 7, 9]
LOCAL_SOURCE_DISTS = {2, 3}

STENCIL_MAX_CAP = 8
STENCIL_MIN_AREA_DIV = 120
STENCIL_MAX_AREA_DIV = 40

BYSTANDER_MAX = 3

CLUSTER_RETRIES = 800
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500

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


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _discover_pattern_components(
    grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR, marker_color: int = MARKER_COLOR
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    mask = (grid == int(pattern_color)) | (grid == int(marker_color))
    comps = components_mask(mask, connectivity=8)
    patterns: list[dict] = []
    cell_to_pattern: dict[tuple[int, int], int] = {}

    for pts in comps:
        teal_cells: list[tuple[int, int]] = []
        marker_cells: list[tuple[int, int]] = []
        for r, c in map(tuple, pts):
            rr = int(r)
            cc = int(c)
            color = int(grid[rr, cc])
            if color == int(pattern_color):
                teal_cells.append((rr, cc))
            elif color == int(marker_color):
                marker_cells.append((rr, cc))
        if len(marker_cells) != 1 or not teal_cells:
            continue

        idx = len(patterns)
        marker = marker_cells[0]
        teal_arr = np.asarray(sorted(teal_cells), dtype=int)
        patterns.append({"marker": marker, "teal": teal_arr})
        for cell in teal_cells:
            cell_to_pattern[(int(cell[0]), int(cell[1]))] = idx
        cell_to_pattern[(int(marker[0]), int(marker[1]))] = idx

    return patterns, cell_to_pattern


def _discover_legends(
    grid: np.ndarray,
    bg: int,
    pattern_color: int = PATTERN_COLOR,
    marker_color: int = MARKER_COLOR,
) -> list[dict]:
    h, w = grid.shape
    patterns, cell_to_pattern = _discover_pattern_components(grid, bg, pattern_color, marker_color)
    legends: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color), int(marker_color)):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                pr = tr + dr
                pc = tc + dc
                if not (0 <= tr < h and 0 <= tc < w and 0 <= pr < h and 0 <= pc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color), int(marker_color)) or target_color == source_color:
                    continue

                pattern_index = cell_to_pattern.get((int(pr), int(pc)))
                if pattern_index is None or int(grid[pr, pc]) != int(pattern_color):
                    continue

                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)

                pattern = patterns[pattern_index]
                legends.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "probe_pos": (int(pr), int(pc)),
                        "anchor": (int(pattern["marker"][0]), int(pattern["marker"][1])),
                        "pattern": np.asarray(pattern["teal"], dtype=int),
                    }
                )

    legends.sort(key=lambda item: (item["source_pos"], item["target_pos"], item["anchor"]))
    return legends


def _assign_local_sources(
    grid: np.ndarray,
    legends: list[dict],
    bg: int,
    pattern_color: int = PATTERN_COLOR,
    marker_color: int = MARKER_COLOR,
) -> tuple[dict[int, tuple[int, int]], set[tuple[int, int]]]:
    legend_pixels: set[tuple[int, int]] = set()
    legends_by_color: dict[int, list[int]] = {}
    for i, legend in enumerate(legends):
        legend_pixels.add(legend["source_pos"])
        legend_pixels.add(legend["target_pos"])
        legends_by_color.setdefault(int(legend["source_color"]), []).append(i)

    positions_by_color: dict[int, list[tuple[int, int]]] = {}
    keep_mask = (grid != int(bg)) & (grid != int(pattern_color)) & (grid != int(marker_color))
    for r, c in map(tuple, np.argwhere(keep_mask)):
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

            anchor_gap = abs(int(pos[0]) - int(legend["anchor"][0])) + abs(int(pos[1]) - int(legend["anchor"][1]))
            candidates.append((dist, anchor_gap, int(pos[0]), int(pos[1])))

        if candidates:
            candidates.sort()
            chosen = (int(candidates[0][2]), int(candidates[0][3]))
            assignments[i] = chosen
            used_positions.add(chosen)

    return assignments, legend_pixels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use nearby source-target-template clusters to transform only their local source pixels.

    Input structure:
    - Black background with several compact clusters and optional unrelated bystanders.
    - Each cluster contains a 4-neighbor source->target legend, a nearby mixed template
      made of teal cells plus one blue marker cell, and one extra source-colored pixel
      2-3 cells away that serves as the execution site for that local legend.
    - The legend's continuation cell after the target touches a teal cell of the template,
      but the true anchor is the blue marker inside that same component.

    Rule:
    - For each legend, find the touched mixed template, keep only its teal cells as the
      stencil, and treat the blue marker as the alignment anchor.
    - Apply that stencil only to the nearby same-colored execution pixel uniquely paired
      with the legend, recoloring the copied teal cells to the legend's target color.
    - Remove all legend pixels, all mixed-template pixels, and the execution pixels that
      were actually claimed by a local legend.

    Key insight: Two different disambiguations must be combined. Proximity chooses which
    source pixel a legend controls, while the embedded marker chooses how the stencil is
    shifted when it is copied.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    legends = _discover_legends(g, bg=bg, pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR)
    assignments, legend_pixels = _assign_local_sources(
        g, legends, bg=bg, pattern_color=PATTERN_COLOR, marker_color=MARKER_COLOR
    )
    local_source_pixels = set(assignments.values())

    out = np.full_like(g, int(bg))

    keep_mask = (g != int(bg)) & (g != int(PATTERN_COLOR)) & (g != int(MARKER_COLOR))
    for r, c in map(tuple, np.argwhere(keep_mask)):
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


def _expand_with_moat(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((int(r) + dr, int(c) + dc))
    return expanded


def _make_mixed_stencil(min_cells: int, max_cells: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    n = random.randint(min_cells, max_cells)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))

    teal_rel = sorted(p for p in coords if p != (0, 0))
    assert teal_rel, "mixed template must contain at least one teal cell"
    touch_rel = random.choice(teal_rel)
    return teal_rel, touch_rel


def _allocate_counts(num_groups: int, num_families: int) -> list[int]:
    counts = [1] * num_families
    for _ in range(num_groups - num_families):
        counts[random.randrange(num_families)] += 1
    random.shuffle(counts)
    return counts


def _build_groups(num_groups: int, area: int) -> tuple[list[dict], set[int]]:
    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(2, area // STENCIL_MIN_AREA_DIV), max_s)

    max_families = min(3, num_groups - 1)
    num_families = random.randint(1, max_families)
    counts = _allocate_counts(num_groups, num_families)

    colors = PAIR_COLORS[:]
    random.shuffle(colors)
    src_colors = colors[:num_families]
    target_pool_global = [c for c in PAIR_COLORS if c not in src_colors]

    groups: list[dict] = []
    used_colors = {MARKER_COLOR}
    for src_color, count in zip(src_colors, counts):
        targets = random.sample(target_pool_global, count)
        for target_color in targets:
            teal_rel, touch_rel = _make_mixed_stencil(min_s, max_s)
            groups.append(
                {
                    "src": int(src_color),
                    "tgt": int(target_color),
                    "teal_rel": teal_rel,
                    "touch_rel": touch_rel,
                }
            )
            used_colors.add(int(src_color))
            used_colors.add(int(target_color))

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
        placed = False
        for _ in range(CLUSTER_RETRIES):
            dr, dc = random.choice(_DIRS4)
            off_r, off_c = random.choice(_offsets_for_direction(dr, dc))
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            pr, pc = sr + 2 * dr, sc + 2 * dc
            er, ec = sr + off_r, sc + off_c
            anchor_r = pr - group["touch_rel"][0]
            anchor_c = pc - group["touch_rel"][1]

            if not (0 <= tr < h and 0 <= tc < w and 0 <= pr < h and 0 <= pc < w and 0 <= er < h and 0 <= ec < w):
                continue
            if not (0 <= anchor_r < h and 0 <= anchor_c < w):
                continue

            teal_cells: set[tuple[int, int]] = set()
            stamp_cells: set[tuple[int, int]] = set()
            ok = True
            for rr, cc in group["teal_rel"]:
                teal = (anchor_r + rr, anchor_c + cc)
                stamp = (er + rr, ec + cc)
                if not (0 <= teal[0] < h and 0 <= teal[1] < w and 0 <= stamp[0] < h and 0 <= stamp[1] < w):
                    ok = False
                    break
                teal_cells.add(teal)
                stamp_cells.add(stamp)
            if not ok or (pr, pc) not in teal_cells:
                continue

            marker_cell = (anchor_r, anchor_c)
            legend_cells = {(sr, sc), (tr, tc)}
            template_cells = set(teal_cells) | {marker_cell}
            instruction_cells = legend_cells | template_cells
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
            inp[marker_cell] = MARKER_COLOR
            for nr, nc in teal_cells:
                inp[nr, nc] = PATTERN_COLOR

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


def _rebuild_blocked_from_groups(groups: list[dict]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for group in groups:
        blocked.update(_expand_with_moat(set(group["footprint"])))
    return blocked


def _place_bystanders(used_colors: set[int], inp: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in PAIR_COLORS if c not in used_colors]
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
            inp[r, c] = int(color)
            placed.append((int(r), int(c), int(color)))
            blocked.update(_expand_with_moat({(int(r), int(c))}))
            break
    return placed


def _build_output(groups: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for group in groups:
        for r, c in group["stamp"]:
            out[int(r), int(c)] = int(group["tgt"])
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
