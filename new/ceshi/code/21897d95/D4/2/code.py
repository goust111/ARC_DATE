# Variant of 21897d95
# Dimension: D4 — 标记方向判定机制 (src)
# Change: 茎方向为T中心的3个分支中唯一没有对称方向的那个；茎指向目标区域，茎方向+横杆方向的两个对角线单元在源区域中... → 源区域由茎尖端邻居而非对角线确定
#
# Design analysis for the D4 variant
# ==================================
# a) Concrete behavior / trace
#    - The T shape itself is unchanged: the center has three orthogonal branches, and the unique
#      branch without an opposite mate is still the stem.
#    - Destination lookup is unchanged: walk from the center along the stem; the first non-control
#      base component on that ray is the region to repaint.
#    - The changed part is the source probe:
#      old: source cells were the two diagonals at center + stem + bar_dir.
#      new: source cells are the two lateral neighbors of the stem tip at distance 2 from center,
#           i.e. center + 2*stem + bar_dir.
#    - Small trace with stem=RIGHT:
#      old source probes: (r-1,c+1) and (r+1,c+1)
#      new source probes: (r-1,c+2) and (r+1,c+2)
#      The solver must therefore "look through" the marker to read the source color.
#    - The generator must place Ts where the c+2 cell lies in the destination, while the two cells
#      above and below that stem tip belong to the source region.
#
# b) Complexity calibration
#    - The variant keeps the original object inventory and rotation rule, so the task does not
#      explode in freedom.
#    - The source probe now depends on a second ring around the marker, which is visually distinct
#      from the old diagonal rule and therefore observable across few examples.
#    - To avoid accidental regression to the old behavior, generation rejects placements where the
#      old diagonal probes would still both lie in the source region.
#
# c) Old assumptions that break
#    - In `_solve_unrotated`, the original `source_parts` loop assumes the old diagonal positions
#      are the source witnesses. That is no longer valid; those cells can now belong to destination
#      or a third region.
#    - The original generator's `_find_valid_placements` enforced the same old diagonals in `src`.
#      That would silently keep producing original-style examples, so it must be rewritten.
#    - The rest of the pipeline (marker parsing, simultaneous recolor, colored-center override,
#      marker repainting, optional L9 rotation) does not depend on the old source geometry.
#
# d) Placement legality
#    - All 4 T cells still must stay in-bounds and non-overlapping.
#    - The distance-2 stem tip cell must stay free and belong to `dst`.
#    - The two lateral neighbors of that stem tip must both lie in `src`; this guarantees the new
#      source readout is well-defined and that solver assertions hold.
#    - L9 placement still avoids marker cells and their protected neighborhood, so it cannot clobber
#      the stem-tip probes after the T markers are placed.
#
# e) Visual observability
#    - A human can see that the recolored destination is not explained by the near diagonals alone:
#      the matching source color sits one step farther out, across the T stem.
#    - Samples are generated with multiple Ts, so the repeated "look past the stem, then sideways"
#      pattern is reinforced inside each grid.
#
# f) Visual cleanliness
#    - The marker silhouette is unchanged and easy to recognize.
#    - The only extra structure is a source-colored wraparound near the stem tip; because regions are
#      large monochrome pieces, this remains legible rather than noisy.
#
# g) Color inference
#    - Colors are still direct copies from a visible source region or from a colored T center.
#    - No invisible ordering or color-id arithmetic is introduced.
#
# h) New solver assumptions and generator guarantees
#    - Every parsed T must have a reachable destination component on its stem ray.
#      Generator guarantee: the distance-2 stem-tip cell is explicitly placed in `dst`.
#    - The two stem-tip lateral probes must belong to same-color source components.
#      Generator guarantee: both probe cells are required to be in the same source region id.
#    - The old diagonals are allowed to disagree with the new source and often do; this is intended.

from __future__ import annotations

from collections import Counter
import sys
import numpy as np
import random

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
RIGHT = (0, 1)


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


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-int(d[0]), -int(d[1]))


def _stem_from_branches(branches: list[tuple[int, int]]) -> tuple[int, int]:
    assert len(branches) == 3
    branch_set = set(branches)
    for d in branches:
        if _opp(d) not in branch_set:
            return d
    raise AssertionError("T marker has no single stem")


def _parse_tetromino(
    cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    if len(cells) != 4:
        return None
    for r, c in cells:
        branches = [(dr, dc) for dr, dc in DIRS if (r + dr, c + dc) in cells]
        if len(branches) != 3:
            continue
        stem = _stem_from_branches(branches)
        expected = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        expected.remove((r - stem[0], c - stem[1]))
        if cells == expected:
            bar = [d for d in branches if d != stem]
            assert len(bar) == 2 and bar[0] == _opp(bar[1])
            return (int(r), int(c)), stem, bar
    return None


def _source_probe_cells(center: tuple[int, int], stem: tuple[int, int], bar: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the two new source witness cells beside the stem tip at distance 2."""
    r, c = center
    return [(r + 2 * stem[0] + bd[0], c + 2 * stem[1] + bd[1]) for bd in bar]


def _legacy_diagonal_cells(center: tuple[int, int], stem: tuple[int, int], bar: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Old-source witness cells, kept only for non-degeneracy checks in the generator/tests."""
    r, c = center
    return [(r + stem[0] + bd[0], c + stem[1] + bd[1]) for bd in bar]


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
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        markers.append({"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells})
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            branches: list[tuple[int, int]] = []
            cells: list[tuple[int, int]] = []
            ok = True
            for d in DIRS:
                rr, cc = r + d[0], c + d[1]
                if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != 1:
                    continue
                label = int(labels_1[rr, cc])
                if one_sizes[label] == 1 and (rr, cc) not in used_ones:
                    branches.append(d)
                    cells.append((rr, cc))
                else:
                    ok = False
            if ok and len(branches) == 3:
                stem = _stem_from_branches(branches)
                bar = [d for d in branches if d != stem]
                assert len(bar) == 2 and bar[0] == _opp(bar[1])
                marker_cells = set(cells) | {(int(r), int(c))}
                markers.append(
                    {
                        "kind": "colored",
                        "center": (int(r), int(c)),
                        "stem": stem,
                        "bar": bar,
                        "cells": marker_cells,
                        "target_color": int(g[r, c]),
                    }
                )
                used_ones |= set(cells)
                for rr, cc in marker_cells:
                    control[rr, cc] = True

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a T control marker"

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

    for marker in markers:
        if marker["kind"] in {"normal", "colored"}:
            center = marker["center"]
            stem = marker["stem"]
            bar = marker["bar"]
            dest = first_component_from(center, stem, start=2)
            assert dest is not None, "T marker must point at a destination region"

            source_parts: list[int] = []
            for rr, cc in _source_probe_cells(center, stem, bar):
                assert _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0
                source_parts.append(int(comp_id[rr, cc]))
            source = source_parts[0]
            for part in source_parts[1:]:
                assert comp_colors[part] == comp_colors[source], "stem-tip source probes must share a color"
                dsu.union(source, part)

            for comp in adjacent_component_counts(marker["cells"]):
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
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
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting T controls for one region"
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
    Repaint regions from T-shaped control markers, erase the markers, and optionally rotate the result using a color-9 L marker.

    Input structure:
    - The canvas is partitioned into monochrome regions.
    - Color 1 forms T controls on region boundaries.
    - Some controls have a colored center with three singleton color-1 branches; that center gives a literal replacement color.
    - An optional 5-cell L of color 9 signals a 90-degree rotation of the final output.

    Rule:
    - The stem is the unique branch direction that has no opposite mate in the T.
    - The stem points toward the destination region, found by moving outward from the center along the stem.
    - The source region is not read from the near diagonals; instead, it is read from the two cells beside the stem tip two steps away from the center.
    - Normal T markers copy the source region's original color into the destination; colored-center markers use the center color instead.
    - Every marker cell is repainted with the final color of the source-side region, and an L9 marker rotates the finished grid: ┐ means CCW 90 degrees, ┌ means CW 90 degrees.

    Key insight:
    - To identify the color donor, follow the stem through the marker to its outer tip and then inspect the two sideways neighbors there.
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

# ── Tunable constants ──────────────────────────────────────────────────
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

COLOR_POOL = [c for c in range(10) if c not in {1, 9}]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 400

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MARKER_COUNTS = [4]
AMBIENT_BAND_COUNTS = [2]


def _get_bar(stem):
    """Return the two bar directions perpendicular to *stem*."""
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center, stem):
    """Return the 4 cells of a T marker: center + stem tip + 2 bar cells."""
    r, c = center
    bar = _get_bar(stem)
    return {
        (r, c),
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_branch_cells(center, stem):
    """Return the 3 branch cells (stem tip + 2 bar), excluding the center."""
    r, c = center
    bar = _get_bar(stem)
    return {
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_neighborhood(center, stem):
    """Return cells adjacent to any T cell but not part of the T itself."""
    cells = _t_cells(center, stem)
    neighborhood = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _island_shape(stem):
    """Connected destination shape whose local geometry differs from the legacy diagonal rule."""
    if stem == RIGHT:
        return {
            (0, 2), (0, 3),
            (-1, 4), (0, 4), (1, 4),
            (-1, 5), (0, 5), (1, 5),
        }
    if stem == (0, -1):
        return {
            (0, -2), (0, -3),
            (-1, -4), (0, -4), (1, -4),
            (-1, -5), (0, -5), (1, -5),
        }
    raise AssertionError("generator only uses horizontal stems")


def _paint_island(region: np.ndarray, ambient_id: int, dst_id: int, center: tuple[int, int], stem: tuple[int, int]) -> bool:
    cells = {(center[0] + dr, center[1] + dc) for dr, dc in _island_shape(stem)}
    H, W = region.shape
    if not all(0 <= r < H and 0 <= c < W for r, c in cells):
        return False
    if not all(int(region[r, c]) == ambient_id for r, c in cells):
        return False
    for r, c in cells:
        region[r, c] = dst_id
    return True


def _sample_band_centers(r0: int, r1: int, count: int) -> list[int]:
    """Choose center rows with at least 4 cells of vertical spacing inside [r0, r1]."""
    centers = []
    cursor = r0 + 2
    for i in range(count):
        remaining = count - i - 1
        hi = r1 - 2 - 4 * remaining
        if cursor > hi:
            raise ValueError("band too small for requested markers")
        center = random.randint(cursor, hi)
        centers.append(center)
        cursor = center + 4
    return centers


def _build_structured_region_map() -> tuple[np.ndarray, list[tuple[int, int, tuple[int, int], tuple[int, int]]]] | None:
    """
    Build two horizontal bands, each split into a host slab and a source slab.

    Band 0 uses RIGHT-pointing markers:
    host on the left, source on the right, destination islands embedded in the right slab.
    Band 1 uses LEFT-pointing markers:
    source on the left, host on the right, destination islands embedded in the left slab.
    """
    n_markers = random.choice(MARKER_COUNTS)
    n_bands = random.choice(AMBIENT_BAND_COUNTS)
    markers_per_band = [n_markers // n_bands] * n_bands
    for i in range(n_markers % n_bands):
        markers_per_band[i] += 1

    band_mins = [4 * count + 3 for count in markers_per_band]
    min_h = sum(band_mins)
    if min_h > GRID_SIZE_MAX:
        return None
    H = random.randint(max(16, min_h), GRID_SIZE_MAX)
    W = random.randint(20, GRID_SIZE_MAX)

    extras = H - min_h
    band_heights = list(band_mins)
    for _ in range(extras):
        band_heights[random.randrange(n_bands)] += 1

    if W < 16:
        return None

    region = np.full((H, W), -1, dtype=int)
    band_ranges: list[tuple[int, int]] = []
    row = 0
    base_layouts = []
    next_rid = 0
    for band_idx, height in enumerate(band_heights):
        split = random.randint(8, W - 8)
        if band_idx == 0:
            host_id, source_id = next_rid, next_rid + 1
            region[row:row + height, :split] = host_id
            region[row:row + height, split:] = source_id
            base_layouts.append((host_id, source_id, split, RIGHT))
        else:
            source_id, host_id = next_rid, next_rid + 1
            region[row:row + height, :split] = source_id
            region[row:row + height, split:] = host_id
            base_layouts.append((host_id, source_id, split, (0, -1)))
        band_ranges.append((row, row + height - 1))
        row += height
        next_rid += 2

    marker_defs: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    dst_id = next_rid
    for band_idx, count in enumerate(markers_per_band):
        r0, r1 = band_ranges[band_idx]
        host_id, source_id, split, stem = base_layouts[band_idx]
        try:
            centers_r = _sample_band_centers(r0, r1, count)
        except ValueError:
            return None
        for center_r in centers_r:
            center_c = split - 2 if stem == RIGHT else split + 1
            center = (center_r, center_c)
            if not _paint_island(region, source_id, dst_id, center, stem):
                return None
            marker_defs.append((source_id, dst_id, center, stem))
            dst_id += 1

    for rid in range(dst_id):
        mask = region == rid
        if mask.sum() < MIN_REGION_CELLS:
            return None
        _, n_cc = label_mask(mask, connectivity=4)
        if n_cc != 1:
            return None
    return region, marker_defs


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    """Grow n_regions from random seeds via randomized BFS; returns region-id array or None."""
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
    """Return all directed (src, dst) pairs between adjacent regions."""
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
    """Return (n_markers, n_normal, n_colored) or None if infeasible."""
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
    """Find all valid (center, stem) for placing a T marker from src toward dst."""
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in _DIRS:
                bar = _get_bar(stem)
                center = (r, c)
                cells = _t_cells(center, stem)
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                if cells & occupied_cells:
                    continue
                if _t_neighborhood(center, stem) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue

                tip = (r + 2 * stem[0], c + 2 * stem[1])
                if not (0 <= tip[0] < H and 0 <= tip[1] < W):
                    continue
                if tip in occupied_cells or tip in occupied_neighbors:
                    continue
                if int(region[tip]) != dst:
                    continue

                source_probe = _source_probe_cells(center, stem, bar)
                if not all(0 <= sr < H and 0 <= sc < W for sr, sc in source_probe):
                    continue
                if any((sr, sc) in occupied_cells for sr, sc in source_probe):
                    continue
                if not all(int(region[sr, sc]) == src for sr, sc in source_probe):
                    continue

                legacy_probe = _legacy_diagonal_cells(center, stem, bar)
                legacy_src_hits = sum(int(region[lr, lc]) == src for lr, lc in legacy_probe)
                if legacy_src_hits == len(legacy_probe):
                    continue

                candidates.append((center, stem))
    return candidates


def _collect_placement_candidates(region, occupied_cells, occupied_neighbors, H, W, dest_used):
    """Enumerate all currently placeable T markers keyed by (src, dst)."""
    candidates = []
    for src, dst in _find_adjacent_pairs(region):
        if dst in dest_used:
            continue
        for center, stem in _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W):
            candidates.append((src, dst, center, stem))
    return candidates


def _register_marker(kind, center, stem, occupied_cells, occupied_neighbors):
    """Update occupied_cells and occupied_neighbors after placing a marker."""
    cells = _t_cells(center, stem)
    if kind == "normal":
        occupied_cells |= cells
        occupied_neighbors |= (_t_neighborhood(center, stem) - cells)
    else:
        branch = _t_branch_cells(center, stem)
        occupied_cells |= branch
        occupied_cells.add(center)
        for br, bc in branch:
            for dr, dc in _DIRS:
                nb = (br + dr, bc + dc)
                if nb != center and nb not in branch:
                    occupied_neighbors.add(nb)
        for dr, dc in _DIRS:
            nb = (center[0] + dr, center[1] + dc)
            if nb not in branch:
                occupied_neighbors.add(nb)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    """Map region ids to colors via vectorized lookup."""
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid, placed_markers):
    """Write T-marker cells onto the input grid."""
    for kind, center, stem, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = tc
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(grid, placed_markers, out_colors):
    """Fill marker cells in the output grid with the source region's final color."""
    for kind, center, stem, src, _dst, _tc in placed_markers:
        color = out_colors[src]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors):
    """Attempt to place a 5-cell L9 marker; return orientation string or None."""
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for kind, center, stem, *_ in placed_markers:
        all_marker_cells |= _t_cells(center, stem)

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
    built = _build_structured_region_map()
    if built is None:
        return None
    region, base_markers = built
    H, W = region.shape
    n_regions = int(region.max()) + 1
    n_markers = len(base_markers)
    if n_markers < MARKER_MIN:
        return None

    counts = _choose_marker_counts(max(n_regions, n_markers))
    if counts is None:
        return None
    _, _n_normal_hint, n_colored = counts
    n_colored = min(n_colored, 2, max(0, n_markers - 2))
    n_normal = n_markers - n_colored

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    random.shuffle(base_markers)
    normal_base = base_markers[:n_normal]
    colored_base = base_markers[n_normal:]

    out_colors = list(input_colors)
    colored_targets = []
    for _src, dst, _center, _stem in colored_base:
        cands = [c for c in pool if c != input_colors[dst]]
        if not cands:
            return None
        colored_targets.append(random.choice(cands))

    for src, dst, _center, _stem in normal_base:
        out_colors[dst] = input_colors[src]
    for (src, dst, _center, _stem), tc in zip(colored_base, colored_targets):
        out_colors[dst] = tc

    placed_markers = [("normal", center, stem, src, dst, None) for src, dst, center, stem in normal_base] + [
        ("colored", center, stem, src, dst, tc)
        for (src, dst, center, stem), tc in zip(colored_base, colored_targets)
    ]

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    for kind, center, stem, _src, _dst, _tc in placed_markers:
        _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)

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
