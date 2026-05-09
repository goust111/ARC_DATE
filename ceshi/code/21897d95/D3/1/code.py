# Variant of 21897d95
# Dimension: D3 — T标记的形状与结构 (src)
# Change: 4单元T-四连骨牌形状，中心有3个邻居，茎为唯一没有对称方向的分支；彩色T中心为非1颜色+3个单独颜色1邻居... → 十字形标记（5单元）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Old normal marker: a 4-cell T of color 1. Its geometry itself reveals the stem.
#    - New normal marker: a full 5-cell cross '+' of color 1.
#    - Old colored marker: colored center + 3 singleton color-1 arms.
#    - New colored marker: colored center + 4 singleton color-1 arms, again a full '+' footprint.
#    - Because the cross is shape-symmetric, the solver cannot read the destination direction from
#      a protruding arm. Instead it infers a "gap side" from context:
#        * for each candidate stem d, inspect the two diagonal cells one step beyond that side;
#        * those two cells still identify the source region, as in the original task;
#        * the opposite arm must be the unique arm that does not touch the source at all;
#        * that source-free opposite arm is the gap side, and its opposite arm is the stem.
#      Example with stem=RIGHT:
#        old logic: the east arm is the unique unmatched branch, so stem=RIGHT immediately.
#        new logic: the 5-cell '+' has all four arms; the east-side diagonals
#        (north-east and south-east of the center) identify the source, and the new west arm is the
#        only arm with zero source contact, so the stem is inferred as the arm opposite that gap.
# b) Complexity calibration
#    - The marker is visually different from the original because every normal control is a 5-cell '+'
#      and every colored control has four color-1 satellites, not three.
#    - To avoid ambiguity, generation enforces a unique gap side for every cross. Relative to the
#      chosen source, exactly one arm may have zero source contact, and it must be opposite the stem.
# c) Implicit-assumption review of the old solver
#    - Parsing old 4-cell components (`_parse_tetromino`) no longer applies; now size-5 color-1
#      components are parsed as full crosses with no built-in orientation.
#    - Colored controls used to require exactly three singleton color-1 neighbors; now they require
#      all four cardinal singleton neighbors.
#    - Old stem inference depended on the branch set itself. That assumption is invalid for a full
#      cross, so orientation is inferred only after base components are labeled.
#    - The destination walk still starts at step 2 because step 1 is the cross arm.
# d) Placement legality
#    - Generator places markers only where the 5-cell cross fits in-bounds, does not overlap other
#      markers, and keeps a buffer so color-1 components do not merge accidentally.
#    - It also enforces the unique-gap invariant, guaranteeing the solver can infer the
#      same orientation it was generated with.
# e) Visual observability
#    - From 2-4 examples, a human can now notice that controls are plus signs, not T tetrominoes.
#    - Direction is recovered from which arm of the cross lacks any source contact, so the task
#      still has a single repeated structural cue rather than arbitrary cases.
# f) Visual cleanliness
#    - Controls are buffered from each other and from the optional L-marker, avoiding merged glyphs.
#    - Colored-center crosses keep each arm singleton so they remain legible as a cross rather than a
#      noisy region fragment.
# g) Color-rule inference
#    - Recoloring logic is unchanged apart from the marker shape: normal crosses copy the source
#      region color, colored-center crosses use the center color, and marker cells are filled with
#      the source side's final color.


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


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-int(d[0]), -int(d[1]))


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if int(stem[0]) == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _cross_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(int(r), int(c))} | {(int(r + dr), int(c + dc)) for dr, dc in DIRS}


def _cross_arm_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(int(r + dr), int(c + dc)) for dr, dc in DIRS}


def _cross_neighborhood(center: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _cross_cells(center)
    neighborhood: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (int(r + dr), int(c + dc))
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _diag_cells(center: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [
        (int(r + direction[0] + br[0]), int(c + direction[1] + br[1]))
        for br in _get_bar(direction)
    ]


def _gap_cells(center: tuple[int, int], stem: tuple[int, int]) -> list[tuple[int, int]]:
    return _diag_cells(center, _opp(stem))


def _parse_cross(cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    if len(cells) != 5:
        return None
    for r, c in cells:
        if all((int(r + dr), int(c + dc)) in cells for dr, dc in DIRS):
            if cells == _cross_cells((int(r), int(c))):
                return int(r), int(c)
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


def _supported_diag_parts(
    center: tuple[int, int], direction: tuple[int, int], comp_id: np.ndarray, comp_colors: list[int]
) -> list[int] | None:
    parts: list[int] = []
    for rr, cc in _diag_cells(center, direction):
        if not _in_bounds(comp_id.shape, rr, cc):
            return None
        cid = int(comp_id[rr, cc])
        if cid < 0:
            return None
        parts.append(cid)
    if comp_colors[parts[0]] != comp_colors[parts[1]]:
        return None
    return parts


def _arm_touches_color(
    center: tuple[int, int], direction: tuple[int, int], comp_id: np.ndarray, comp_colors: list[int], color: int
) -> bool:
    ar, ac = center[0] + direction[0], center[1] + direction[1]
    for dr, dc in DIRS:
        rr, cc = ar + dr, ac + dc
        if (rr, cc) == center or not _in_bounds(comp_id.shape, rr, cc):
            continue
        cid = int(comp_id[rr, cc])
        if cid >= 0 and comp_colors[cid] == int(color):
            return True
    return False


def _infer_cross_orientation(
    center: tuple[int, int], comp_id: np.ndarray, comp_colors: list[int]
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], list[int]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int], list[int]]] = []
    for stem in DIRS:
        source_parts = _supported_diag_parts(center, stem, comp_id, comp_colors)
        if source_parts is None:
            continue
        source_color = int(comp_colors[source_parts[0]])
        gap = _opp(stem)
        if _arm_touches_color(center, gap, comp_id, comp_colors, source_color):
            continue
        if not all(
            _arm_touches_color(center, direction, comp_id, comp_colors, source_color)
            for direction in DIRS
            if direction != gap
        ):
            continue
        candidates.append((gap, stem, source_parts))

    assert len(candidates) == 1, "cross marker must have one unique gap side"
    gap, stem, source_parts = candidates[0]
    return gap, stem, _get_bar(stem), source_parts


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
        center = _parse_cross(cells)
        if center is None:
            continue
        markers.append({"kind": "normal", "center": center, "cells": cells})
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            arms: list[tuple[int, int]] = []
            ok = True
            for d in DIRS:
                rr, cc = r + d[0], c + d[1]
                if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != 1:
                    ok = False
                    break
                label = int(labels_1[rr, cc])
                if one_sizes[label] != 1 or (rr, cc) in used_ones:
                    ok = False
                    break
                arms.append((int(rr), int(cc)))
            if ok and len(arms) == 4:
                marker_cells = set(arms) | {(int(r), int(c))}
                markers.append(
                    {
                        "kind": "colored",
                        "center": (int(r), int(c)),
                        "cells": marker_cells,
                        "target_color": int(g[r, c]),
                    }
                )
                used_ones |= set(arms)
                for rr, cc in marker_cells:
                    control[rr, cc] = True

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a cross control marker"

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
            gap, stem, bar, source_parts = _infer_cross_orientation(center, comp_id, comp_colors)
            dest = first_component_from(center, stem, start=2)
            assert dest is not None, "cross marker must point at a destination region"

            source = int(source_parts[0])
            for part in source_parts[1:]:
                assert comp_colors[part] == comp_colors[source], "source diagonals must share a color"
                dsu.union(source, part)

            for comp in adjacent_component_counts(marker["cells"]):
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
            marker["gap"] = gap
            marker["stem"] = stem
            marker["bar"] = bar
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
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting cross controls for one region"
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
    Repaint regions from cross-shaped control markers, erase the markers, and optionally rotate based on a color-9 L-marker.

    Input structure:
    - The grid is partitioned into solid-color regions.
    - Normal controls are full 5-cell '+' markers made entirely of color 1.
    - Colored controls are also full '+' markers, but their center is a non-1 literal color and the four cardinal arms are singleton color-1 cells.
    - An optional 5-cell L of color 9 indicates a 90-degree rotation after recoloring.

    Rule:
    - Each cross still acts like a directional recoloring control, but the direction is inferred indirectly.
    - Use the same-region diagonal pair on one side to identify the source region, then find the one arm of the cross that never touches that source; the opposite arm is the stem and points to the destination region.
    - A normal cross repaints the destination to the source region's original color; a colored-center cross repaints it to the center color.
    - Every marker cell is filled with the final color of the source-side region, and the L-shaped color-9 marker still controls optional rotation.

    Key insight:
    - The marker no longer tells its direction by sticking out. Instead, the full '+' is symmetric and you must spot which arm has a local "gap" to the source region; the opposite arm is the stem.
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
# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5]
REGION_WEIGHTS = [3, 2]

# Marker counts
MARKER_MIN = 2
MARKER_MAX = 4
NORMAL_MAX = 4
COLORED_MAX = 2

# Color pool: exclude 1 for marker arms and 9 for the rotation marker.
COLOR_POOL = [c for c in range(10) if c not in {1, 9}]

# L9 rotation marker
L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

# Generation retry budget
SEED_ATTEMPTS = 200
GEN_RETRIES = 300


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
        for dr, dc in DIRS:
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
        for dr, dc in DIRS:
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
            for dr, dc in DIRS:
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


def _has_supported_pair(region: np.ndarray, center: tuple[int, int], direction: tuple[int, int], src: int | None = None) -> bool:
    coords = _diag_cells(center, direction)
    if not all(_in_bounds(region.shape, rr, cc) for rr, cc in coords):
        return False
    vals = [int(region[rr, cc]) for rr, cc in coords]
    if src is not None:
        return vals[0] == vals[1] == int(src)
    return vals[0] == vals[1]


def _arm_touches_region(region: np.ndarray, center: tuple[int, int], direction: tuple[int, int], src: int) -> bool:
    ar, ac = center[0] + direction[0], center[1] + direction[1]
    for dr, dc in DIRS:
        rr, cc = ar + dr, ac + dc
        if (rr, cc) == center or not _in_bounds(region.shape, rr, cc):
            continue
        if int(region[rr, cc]) == int(src):
            return True
    return False


def _gap_replacements(
    region: np.ndarray, center: tuple[int, int], stem: tuple[int, int], src: int
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    cells = _gap_cells(center, stem)
    cross = _cross_cells(center)
    options: list[list[int]] = []
    for r, c in cells:
        ids: set[int] = set()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(region.shape, nr, nc):
                continue
            if (nr, nc) in cross:
                continue
            rid = int(region[nr, nc])
            if rid != int(src):
                ids.add(rid)
        current = int(region[r, c])
        if current != int(src):
            ids.add(current)
        if not ids:
            return None
        options.append(sorted(ids))

    for left in options[0]:
        for right in options[1]:
            if left != right:
                return cells, (int(left), int(right))
    return None


def _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W):
    """Find valid (center, stem) placements plus rewrites that force a unique gap side."""
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            center = (r, c)
            cells = _cross_cells(center)
            if cells & occupied_cells:
                continue
            if _cross_neighborhood(center) & occupied_cells:
                continue
            if cells & occupied_neighbors:
                continue
            for stem in DIRS:
                if not _has_supported_pair(region, center, stem, src=src):
                    continue
                step = 2
                found = False
                while True:
                    nr, nc = r + stem[0] * step, c + stem[1] * step
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied_cells or (nr, nc) in cells:
                        step += 1
                        continue
                    found = int(region[nr, nc]) == dst
                    break
                if found:
                    forced = _gap_replacements(region, center, stem, src)
                    if forced is not None:
                        candidates.append((center, stem, forced))
    return candidates


def _register_marker(kind, center, occupied_cells, occupied_neighbors):
    """Update occupied_cells and occupied_neighbors after placing a marker."""
    cells = _cross_cells(center)
    if kind == "normal":
        occupied_cells |= cells
        occupied_neighbors |= (_cross_neighborhood(center) - cells)
    else:
        arms = _cross_arm_cells(center)
        occupied_cells |= arms
        occupied_cells.add(center)
        for ar, ac in arms:
            for dr, dc in DIRS:
                nb = (int(ar + dr), int(ac + dc))
                if nb != center and nb not in arms:
                    occupied_neighbors.add(nb)
        for nb in _cross_neighborhood(center):
            if nb not in cells:
                occupied_neighbors.add(nb)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    """Map region ids to colors via vectorized lookup."""
    lut = np.array(colors, dtype=int)
    return lut[region]


def _apply_cross_context(
    region: np.ndarray,
    center: tuple[int, int],
    stem: tuple[int, int],
    src: int,
    forced: tuple[list[tuple[int, int]], tuple[int, int]],
) -> None:
    for r, c in _diag_cells(center, stem):
        region[r, c] = int(src)
    cells, values = forced
    for (r, c), rid in zip(cells, values):
        region[r, c] = int(rid)


def _stamp_markers_input(grid, placed_markers):
    """Write cross-marker cells onto the input grid."""
    for kind, center, _stem, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for cr, cc in _cross_cells(center):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = tc
            for cr, cc in _cross_arm_cells(center):
                grid[cr, cc] = 1


def _stamp_markers_output(grid, placed_markers, out_colors):
    """Fill marker cells in the output grid with the source region's final color."""
    for _kind, center, _stem, src, _dst, _tc in placed_markers:
        color = out_colors[src]
        for cr, cc in _cross_cells(center):
            grid[cr, cc] = color


def _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors):
    """Attempt to place a 5-cell L9 marker; return orientation string or None."""
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, _stem, *_ in placed_markers:
        all_marker_cells |= _cross_cells(center)

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
    feasible_pairs: list[tuple[int, int, int]] = []
    for src, dst in directed_pairs:
        count = len(_find_valid_placements(region, src, dst, set(), set(), H, W))
        if count > 0:
            feasible_pairs.append((src, dst, count))

    feasible_destinations = {dst for _src, dst, _count in feasible_pairs}
    if len(feasible_destinations) < MARKER_MIN:
        return None

    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    n_markers, n_normal, n_colored = counts
    n_markers = min(n_markers, len(feasible_destinations))
    if n_markers < MARKER_MIN:
        return None
    n_colored = min(n_colored, n_markers - 1)
    n_normal = n_markers - n_colored

    random.shuffle(feasible_pairs)
    feasible_pairs.sort(key=lambda item: (-item[2], random.random()))
    dest_used = set()
    marker_pairs: list[tuple[int, int, int]] = []
    for src, dst, count in feasible_pairs:
        if dst in dest_used:
            continue
        marker_pairs.append((src, dst, count))
        dest_used.add(dst)
        if len(marker_pairs) >= n_markers:
            break
    if len(marker_pairs) < MARKER_MIN:
        return None
    n_markers = len(marker_pairs)
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
    for _src, dst, _count in colored_pairs:
        cands = [c for c in pool if c != input_colors[dst]]
        if not cands:
            return None
        colored_targets.append(random.choice(cands))

    for src, dst, _count in normal_pairs:
        out_colors[dst] = input_colors[src]
    for (_src, dst, _count), tc in zip(colored_pairs, colored_targets):
        out_colors[dst] = tc

    all_defs = [("normal", s, d, None, count) for s, d, count in normal_pairs] + [
        ("colored", s, d, tc, count) for (s, d, count), tc in zip(colored_pairs, colored_targets)
    ]
    random.shuffle(all_defs)
    all_defs.sort(key=lambda item: item[4])

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers = []
    for kind, src, dst, tc, _count in all_defs:
        cands = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W)
        if not cands:
            return None
        center, stem, forced = random.choice(cands)
        _apply_cross_context(region, center, stem, src, forced)
        _register_marker(kind, center, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, tc))

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
