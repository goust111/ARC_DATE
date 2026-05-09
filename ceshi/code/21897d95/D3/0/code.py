# Variant of 21897d95
# Dimension: D3 — T标记的形状与结构 (src)
# Change: 4单元T-四连骨牌形状，中心有3个邻居，茎为唯一没有对称方向的分支；彩色T中心为非1颜色+3个单独颜色1邻居... → L形标记替代T形

from __future__ import annotations

# Design analysis for the D3 variant
# ----------------------------------
# spec_like:
#   core_change:
#     original_marker: 4-cell T tetromino, stem readable from the 3-neighbor hub.
#     variant_marker: 3-cell L tromino. The elbow has two orthogonal arms, so geometry
#       alone no longer tells us which arm is the stem. The solver must infer the stem
#       from region context.
#   concrete_shape:
#     normal:
#       cells: elbow E, destination arm E+S, source-side arm E+L
#       colors: all 3 cells are color 1
#     colored:
#       cells: colored elbow E, two singleton color-1 arm cells at E+S and E+L
#       target_color: the elbow color
#     semantics:
#       destination: the first base-region cell found from E by walking further along S
#         starting at step 2
#       source: the diagonal probe E+S+L
#       source_side_guard: E+2L is also forced to stay in the source region during
#         generation so the non-stem interpretation collapses back onto the source side
#         and becomes invalid
#   behavior_trace:
#     original_local_rule:
#       a T with stem right repaints the region to the right; the two diagonal cells
#       behind the stem (up-right/down-right) identify the source color.
#     variant_local_rule:
#       an L with arms (right, up) uses one arm as the stem only if walking two cells
#       farther in that arm hits the destination while the diagonal cell at up-right
#       is in the source. The other arm is rejected because walking farther in that
#       direction stays in the source region, so source==dest and the assignment is
#       not a legal control.
#     visible_difference:
#       the control is now a 3-cell corner rather than a 4-cell fork, and the arm that
#       matters must be inferred from the surrounding regions rather than from vertex degree.
#   complexity_calibration:
#     kept:
#       - simultaneous region recoloring
#       - optional colored controls
#       - optional color-9 L rotation marker
#       - small non-rectangular scraps absorbed by adjacency
#     changed_only_for_D3:
#       - marker parsing
#       - marker placement search
#       - marker neighborhood buffering
#     anti_simplification:
#       - 8 ordered L orientations are sampled (4 stem directions x 2 orthogonal turns)
#       - colored and normal markers both remain present
#       - generator enforces unique arm interpretation so the task is not underconstrained
#   implicit_assumption_audit:
#     old_T_assumptions_that_break:
#       - "one color-1 component of size 4 is a marker" no longer holds
#       - "the hub has degree 3, so the stem is the non-opposite branch" no longer holds
#       - "colored markers have 3 singleton 1-neighbors" no longer holds
#     new_solver_assumptions:
#       - every size-3 color-1 component is an orthogonal L, not a straight triomino
#       - every colored control has exactly two orthogonal singleton color-1 neighbors
#       - among the two possible arm assignments, exactly one yields destination != source
#       - the diagonal probe cell is part of a base region, not another control
#     generator_guarantees:
#       - only orthogonal L markers are placed
#       - the elbow and long-arm cell sit on the source side, the stem arm sits on the
#         destination side, and the diagonal probe lies in the source
#       - the first cell at E+2S belongs to the destination, while E+2L belongs to the
#         source, killing the swapped-arm interpretation
#       - marker buffers preserve singleton color-1 arms for colored controls
#   placement_legality:
#     - markers never overlap each other or the buffered neighborhood of existing markers
#     - all marker cells and the source/destination probe cells stay in bounds
#     - L9 markers are still placed away from controls and read back as small scraps plus rotation
#   visual_observability:
#     - the change is human-visible because the control symbol shrinks from a T fork to a corner
#     - the destination arm is supported by repeated examples in multiple orientations
#     - recoloring remains tied to spatial structure; no hidden ordering or color-id logic is used
#   neatness_and_color_inference:
#     - each region still has a single flat color away from controls
#     - markers are buffered so corners do not fuse into larger ambiguous color-1 blobs
#     - colored elbows remain the only place where the replacement color is introduced directly

import random
import sys
from collections import Counter

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(int(n)))

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _is_orthogonal(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return int(a[0]) * int(b[0]) + int(a[1]) * int(b[1]) == 0


def _other_arm(arms: tuple[tuple[int, int], tuple[int, int]], stem: tuple[int, int]) -> tuple[int, int]:
    return arms[1] if arms[0] == stem else arms[0]


def _parse_l_triomino(
    cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] | None:
    if len(cells) != 3:
        return None
    for r, c in cells:
        arms = tuple((dr, dc) for dr, dc in DIRS if (r + dr, c + dc) in cells)
        if len(arms) != 2 or not _is_orthogonal(arms[0], arms[1]):
            continue
        if cells == {(r, c), (r + arms[0][0], c + arms[0][1]), (r + arms[1][0], c + arms[1][1])}:
            return (int(r), int(c)), arms
    return None


def _label_base_components(g: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
    comp_areas: list[int] = []
    cid = 0
    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == col) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == k)
            if coords.size == 0:
                continue
            for rr, cc in coords:
                comp_id[int(rr), int(cc)] = cid
            comp_colors.append(int(col))
            comp_areas.append(int(len(coords)))
            cid += 1
    return comp_id, comp_colors, comp_areas


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        k: {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == k)}
        for k in range(1, int(n_1) + 1)
    }
    one_sizes = {k: len(cells) for k, cells in one_components.items()}

    markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)

    for cells in one_components.values():
        parsed = _parse_l_triomino(cells)
        if parsed is None:
            continue
        center, arms = parsed
        markers.append({"kind": "normal", "center": center, "arms": arms, "cells": cells})
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            arms: list[tuple[int, int]] = []
            one_cells: list[tuple[int, int]] = []
            ok = True
            for d in DIRS:
                rr, cc = r + d[0], c + d[1]
                if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != 1:
                    continue
                label = int(labels_1[rr, cc])
                if one_sizes[label] == 1 and (rr, cc) not in used_ones:
                    arms.append(d)
                    one_cells.append((rr, cc))
                else:
                    ok = False
            if ok and len(arms) == 2 and _is_orthogonal(arms[0], arms[1]):
                marker_cells = set(one_cells) | {(int(r), int(c))}
                markers.append(
                    {
                        "kind": "colored",
                        "center": (int(r), int(c)),
                        "arms": (arms[0], arms[1]),
                        "cells": marker_cells,
                        "target_color": int(g[r, c]),
                    }
                )
                used_ones |= set(one_cells)
                for rr, cc in marker_cells:
                    control[rr, cc] = True

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to an L control marker"

    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == col) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == k)
            if len(coords) > 5:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if len(coords) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
                continue
            cells = {tuple(map(int, xy)) for xy in coords}
            markers.append({"kind": "small", "cells": cells})
            for rr, cc in cells:
                control[rr, cc] = True

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"
    dsu = _DSU(len(comp_colors))

    def first_component_from(center: tuple[int, int], direction: tuple[int, int], start: int) -> int | None:
        r, c = center
        step = int(start)
        while True:
            rr = r + direction[0] * step
            cc = c + direction[1] * step
            if not _in_bounds(g.shape, rr, cc):
                return None
            if comp_id[rr, cc] >= 0:
                return int(comp_id[rr, cc])
            step += 1

    def adjacent_component_counts(cells: set[tuple[int, int]]) -> Counter:
        counts: Counter = Counter()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0:
                    counts[int(comp_id[rr, cc])] += 1
        return counts

    def resolve_l_assignment(
        center: tuple[int, int],
        arms: tuple[tuple[int, int], tuple[int, int]],
    ) -> tuple[tuple[int, int], tuple[int, int], int, int]:
        options: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
        for stem in arms:
            long_arm = _other_arm(arms, stem)
            sr = center[0] + stem[0] + long_arm[0]
            sc = center[1] + stem[1] + long_arm[1]
            if not _in_bounds(g.shape, sr, sc) or comp_id[sr, sc] < 0:
                continue
            source = int(comp_id[sr, sc])
            dest = first_component_from(center, stem, start=2)
            if dest is None or int(dest) == int(source):
                continue
            options.append((stem, long_arm, source, int(dest)))
        assert len(options) == 1, "L marker must have a unique stem assignment"
        return options[0]

    for marker in markers:
        if marker["kind"] in {"normal", "colored"}:
            center = marker["center"]
            stem, long_arm, source, dest = resolve_l_assignment(center, marker["arms"])
            for comp in adjacent_component_counts(marker["cells"]):
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
            marker["stem"] = stem
            marker["long_arm"] = long_arm
            marker["dest_comp"] = dest
            marker["source_comp"] = source
        else:
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "small marker must touch its owner region"
            owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
            for comp in counts:
                if comp_colors[comp] == comp_colors[owner]:
                    dsu.union(owner, comp)
            marker["source_comp"] = owner

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    final_color = dict(root_color)
    for marker in markers:
        if marker["kind"] not in {"normal", "colored"}:
            marker["source"] = dsu.find(marker["source_comp"])
            continue
        dest = dsu.find(marker["dest_comp"])
        source = dsu.find(marker["source_comp"])
        marker["source"] = source
        replacement = int(marker["target_color"]) if marker["kind"] == "colored" else int(root_color[source])
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting L controls for one region"
        final_color[dest] = replacement

    out = np.full(g.shape, -1, dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[dsu.find(cid)])

    for marker in markers:
        color = int(final_color[marker["source"]])
        for r, c in marker["cells"]:
            out[r, c] = color

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def _find_l9_orientation(g: np.ndarray) -> str | None:
    """Find a 5-cell L-shaped color-9 marker and return its corner position."""
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    for k in range(1, int(n) + 1):
        coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == k)}
        if len(coords) != 5:
            continue
        rows = [r for r, c in coords]
        cols = [c for r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        for cr, cc in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
            h_nb = (cr, cc + 1 if cc == c0 else cc - 1)
            v_nb = (cr + 1 if cr == r0 else cr - 1, cc)
            if h_nb in coords and v_nb in coords:
                if cr == r0 and cc == c1:
                    return "top_right"
                if cr == r0 and cc == c0:
                    return "top_left"
                assert False, f"unobserved L9 corner at ({cr},{cc})"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint regions from L-shaped control markers, erase the markers, and optionally rotate based on a color-9 L-marker.

    Input structure:
    - The grid is a mosaic of monochrome regions.
    - Color 1 forms 3-cell L markers on region boundaries instead of 4-cell T markers.
    - Some controls use a colored elbow with exactly two singleton color-1 neighbors, making a colored L; that elbow color is the literal replacement color.
    - An optional 5-cell L-shaped color-9 marker still triggers a 90-degree rotation of the output.

    Rule:
    - Each L has two orthogonal arms. The arm whose forward ray reaches a different base region is the short stem, and that ray identifies the destination region to repaint.
    - The diagonal cell one step beyond both arms belongs to the source region; a normal L copies that source region's original color, while a colored elbow uses its own color instead.
    - All control cells are recolored with the final color of the source-side region.
    - A 5-cell L-shaped color-9 marker controls rotation: ┐ (corner top-right) -> CCW 90 deg, ┌ (corner top-left) -> CW 90 deg.

    Key insight:
    - Unlike the original T, an L does not advertise its stem by vertex degree, so you must infer which arm is the stem from the surrounding regions and use the diagonal beyond the corner to find the source.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (l9rot) ---------------

# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

# Marker counts
MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

# Color pool (all ARC colors except 1, which is reserved for L-marker cells)
COLOR_POOL = [c for c in range(10) if c != 1]

# L9 rotation marker
L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

# Generation retry budget
SEED_ATTEMPTS = 200
GEN_RETRIES = 300

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _orthogonal_dirs(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _l_cells(center: tuple[int, int], stem: tuple[int, int], long_arm: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r + stem[0], c + stem[1]), (r + long_arm[0], c + long_arm[1])}


def _l_one_cells(center: tuple[int, int], stem: tuple[int, int], long_arm: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + stem[0], c + stem[1]), (r + long_arm[0], c + long_arm[1])}


def _l_neighborhood(center: tuple[int, int], stem: tuple[int, int], long_arm: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _l_cells(center, stem, long_arm)
    neighborhood = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    region = np.full((H, W), -1, dtype=int)
    seeds = set()
    for _ in range(SEED_ATTEMPTS):
        if len(seeds) >= n_regions:
            break
        seeds.add((random.randint(1, H - 2), random.randint(1, W - 2)))
    if len(seeds) < n_regions:
        return None

    frontier = []
    for i, (r, c) in enumerate(list(seeds)):
        region[r, c] = i
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and region[nr, nc] == -1:
                frontier.append((nr, nc, i))
    random.shuffle(frontier)

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if region[r, c] != -1:
            continue
        region[r, c] = rid
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and region[nr, nc] == -1:
                frontier.append((nr, nc, rid))

    for rid in range(n_regions):
        mask = region == rid
        if mask.sum() < MIN_REGION_CELLS:
            return None
        _, n = label_mask(mask, connectivity=4)
        if n != 1:
            return None
    return region


def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    H, W = region.shape
    undirected = set()
    for r in range(H):
        for c in range(W):
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    a, b = int(region[r, c]), int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _choose_marker_counts(n_regions: int) -> tuple[int, int, int] | None:
    max_possible = min(MARKER_MAX, n_regions)
    if max_possible < MARKER_MIN:
        return None
    n_markers = random.randint(MARKER_MIN, max_possible)
    n_colored = random.randint(0, min(COLORED_MAX, n_markers - 2))
    n_normal = n_markers - n_colored
    if n_normal > NORMAL_MAX:
        n_colored = n_markers - NORMAL_MAX
        n_normal = NORMAL_MAX
        if n_colored > COLORED_MAX:
            n_markers = NORMAL_MAX + COLORED_MAX
            n_colored = COLORED_MAX
            n_normal = NORMAL_MAX
    return n_markers, n_normal, n_colored


def _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W):
    """Find all valid (center, stem, long_arm) placements for an ordered L marker."""
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in _DIRS:
                for long_arm in _orthogonal_dirs(stem):
                    cells = _l_cells((r, c), stem, long_arm)
                    if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                        continue
                    if cells & occupied_cells:
                        continue
                    if _l_neighborhood((r, c), stem, long_arm) & occupied_cells:
                        continue
                    if cells & occupied_neighbors:
                        continue

                    sr, sc = r + stem[0] + long_arm[0], c + stem[1] + long_arm[1]
                    dr2, dc2 = r + 2 * stem[0], c + 2 * stem[1]
                    lr2, lc2 = r + 2 * long_arm[0], c + 2 * long_arm[1]
                    if not all(0 <= rr < H and 0 <= cc < W for rr, cc in [(sr, sc), (dr2, dc2), (lr2, lc2)]):
                        continue

                    if int(region[r, c]) != src:
                        continue
                    if int(region[r + long_arm[0], c + long_arm[1]]) != src:
                        continue
                    if int(region[r + stem[0], c + stem[1]]) != dst:
                        continue
                    if int(region[sr, sc]) != src:
                        continue
                    if int(region[dr2, dc2]) != dst:
                        continue
                    if int(region[lr2, lc2]) != src:
                        continue

                    candidates.append(((r, c), stem, long_arm))
    return candidates


def _register_marker(kind, center, stem, long_arm, occupied_cells, occupied_neighbors):
    cells = _l_cells(center, stem, long_arm)
    if kind == "normal":
        occupied_cells |= cells
        occupied_neighbors |= (_l_neighborhood(center, stem, long_arm) - cells)
    else:
        arm_cells = _l_one_cells(center, stem, long_arm)
        occupied_cells |= arm_cells
        occupied_cells.add(center)
        for ar, ac in arm_cells:
            for dr, dc in _DIRS:
                nb = (ar + dr, ac + dc)
                if nb != center and nb not in arm_cells:
                    occupied_neighbors.add(nb)
        for dr, dc in _DIRS:
            nb = (center[0] + dr, center[1] + dc)
            if nb not in arm_cells:
                occupied_neighbors.add(nb)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid, placed_markers):
    for kind, center, stem, long_arm, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for cr, cc in _l_cells(center, stem, long_arm):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = tc
            for cr, cc in _l_one_cells(center, stem, long_arm):
                grid[cr, cc] = 1


def _stamp_markers_output(grid, placed_markers, out_colors):
    for _kind, center, stem, long_arm, src, _dst, _tc in placed_markers:
        color = out_colors[src]
        for cr, cc in _l_cells(center, stem, long_arm):
            grid[cr, cc] = color


def _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors):
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, stem, long_arm, *_ in placed_markers:
        all_marker_cells |= _l_cells(center, stem, long_arm)

    candidates = []
    for r in range(H):
        for c in range(W):
            cells = [(r + dr, c + dc) for dr, dc in offsets]
            if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if any((cr, cc) in all_marker_cells for cr, cc in cells):
                continue
            if any((cr, cc) in occupied_neighbors for cr, cc in cells):
                continue
            candidates.append((r, c, rid))

    if not candidates:
        return None
    lr, lc, rid = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[lr + dr, lc + dc] = 9
        output_grid[lr + dr, lc + dc] = out_colors[rid]
    return orient


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    directed_pairs = _find_adjacent_pairs(region)

    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    n_markers, n_normal, n_colored = counts

    random.shuffle(directed_pairs)
    dest_used = set()
    marker_pairs = []
    for src, dst in directed_pairs:
        if dst not in dest_used:
            marker_pairs.append((src, dst))
            dest_used.add(dst)
            if len(marker_pairs) >= n_markers:
                break
    if len(marker_pairs) < n_markers:
        n_markers = len(marker_pairs)
        if n_markers < MARKER_MIN:
            return None
        n_colored = min(n_colored, n_markers - 1)
        n_normal = n_markers - n_colored
    random.shuffle(marker_pairs)

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    out_colors = list(input_colors)
    normal_pairs = marker_pairs[:n_normal]
    colored_pairs = marker_pairs[n_normal:]

    colored_targets = []
    for _src, dst in colored_pairs:
        cands = [c for c in pool if c != input_colors[dst]]
        if not cands:
            return None
        colored_targets.append(random.choice(cands))

    for src, dst in normal_pairs:
        out_colors[dst] = input_colors[src]
    for (_src, dst), tc in zip(colored_pairs, colored_targets):
        out_colors[dst] = tc

    all_defs = [("normal", s, d, None) for s, d in normal_pairs] + [
        ("colored", s, d, tc) for (s, d), tc in zip(colored_pairs, colored_targets)
    ]
    random.shuffle(all_defs)

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers = []
    for kind, src, dst, tc in all_defs:
        cands = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W)
        if not cands:
            return None
        center, stem, long_arm = random.choice(cands)
        _register_marker(kind, center, stem, long_arm, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, long_arm, src, dst, tc))

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    l9_orient = _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors)
    if l9_orient == "top_right":
        output_grid = np.rot90(output_grid, 1)
    elif l9_orient == "top_left":
        output_grid = np.rot90(output_grid, 3)

    try:
        result = main(input_grid)
    except Exception:
        return None
    return input_grid if np.array_equal(result, output_grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
