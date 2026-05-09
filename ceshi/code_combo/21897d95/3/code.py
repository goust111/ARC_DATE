# Combo variant of 21897d95 (index 3)
# combo_spec: D3/0+D4/2
# Name: L形射线寻源

from __future__ import annotations

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior / trace
#    - Original task: a 4-cell T advertises its stem by degree, then reads the source from
#      diagonal cells near the stem.
#    - Combo task: the control is a 3-cell orthogonal L. The elbow has exactly two arms, so
#      geometry alone no longer tells us which arm is the stem. The solver must test both arms:
#      choose the arm whose forward ray reaches a different base region at distance 2, then read
#      the source from the two lateral neighbors of that ray tip.
#    - Example with elbow E=(r,c), candidate stem RIGHT:
#      marker cells = {(r,c), (r,c+1), (r-1,c)} for an upper-turn L.
#      destination witness = (r,c+2).
#      source witnesses = (r-1,c+2) and (r+1,c+2).
#      The old D3-only rule would have read (r-1,c+1); this combo deliberately does not.
#
# b) Complexity calibration
#    - The visible change is strong: the marker shrinks from a T to an L, and the color donor is
#      no longer adjacent to the elbow but sits around the stem tip.
#    - To keep the task learnable from few examples, the generator uses repeated band layouts with
#      several L markers, both left- and right-pointing stems, and both upper/lower turns.
#    - To avoid underconstraint, the generator rejects the swapped-arm interpretation by ensuring
#      the non-stem ray stays in the source region.
#
# c) Implicit assumption audit
#    - Any code assuming a size-4 color-1 component is a marker is invalid; normal controls are
#      now size-3 L triominoes.
#    - Any code assuming a 3-branch hub reveals the stem is invalid; the solver must resolve the
#      stem by testing both arms against surrounding regions.
#    - Any code assuming a colored marker has three singleton color-1 neighbors is invalid; it now
#      has exactly two orthogonal singleton color-1 neighbors.
#    - Any code reading a single diagonal source witness is invalid; the combo reads two lateral
#      witnesses around the stem tip and requires them to agree in color.
#
# d) Placement legality
#    - Marker cells stay in bounds, do not overlap, and are buffered so colored-arm cells remain
#      singleton 1-components.
#    - Each destination island is connected, at least 6 cells, and shaped so the stem ray reaches
#      destination cells at distance 1 and 2 while the two lateral stem-tip cells remain in the
#      source band.
#    - The optional color-9 rotation marker is placed away from marker cells and their buffers.
#
# e) Visual observability
#    - Across examples, the recolored destination can be explained only by following the short arm
#      through the marker and then looking sideways at the tip; the old near-diagonal cell usually
#      belongs to the destination instead.
#    - The left/right repetition lets a human infer that "the arm pointing across the boundary is
#      the one that matters", while the upper/lower turns show that the corner orientation alone
#      does not determine the donor.
#
# f) Visual cleanliness
#    - Large source bands keep the local source-wrap pattern readable.
#    - Destination islands are spaced vertically so markers do not crowd or merge.
#    - Colored elbows remain the only direct introduction of a replacement color.
#
# g) Color inference
#    - Normal controls still copy a visible region color.
#    - Colored controls still use the elbow color literally.
#    - No ordering by color id or hidden metadata is introduced.
#
# h) Quantified anti-degeneracy target
#    - The generator is built so each marker's legacy D3 diagonal probe sits in the destination
#      island, not in the true source band. Therefore the combo output should usually disagree with
#      a legacy diagonal-reading L solver on the same input.
#
# i) New solver assumptions and generator guarantees
#    - Solver assumption: every parsed L marker has exactly one arm assignment whose ray reaches a
#      destination component and whose stem-tip lateral witnesses are valid source cells.
#      Generator guarantee: each destination island is carved so the intended stem is valid and the
#      swapped arm keeps source==destination.
#    - Solver assumption: the two stem-tip witnesses share a source color.
#      Generator guarantee: both witnesses lie in the same source band.
#    - Generator-specific modeling note: the ambient background color is allowed to appear in
#      multiple disconnected components after the source bands are carved. This does not create
#      solver ambiguity because the solver labels connected components per color before reasoning.

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}


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


def _orthogonal_dirs(d: tuple[int, int]) -> list[tuple[int, int]]:
    if d[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


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
        expected = {(r, c), (r + arms[0][0], c + arms[0][1]), (r + arms[1][0], c + arms[1][1])}
        if cells == expected:
            return (int(r), int(c)), arms
    return None


def _tip_lateral_cells(center: tuple[int, int], stem: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + 2 * stem[0] + pr, c + 2 * stem[1] + pc) for pr, pc in _orthogonal_dirs(stem)]


def _legacy_diagonal_cell(
    center: tuple[int, int],
    stem: tuple[int, int],
    long_arm: tuple[int, int],
) -> tuple[int, int]:
    return (center[0] + stem[0] + long_arm[0], center[1] + stem[1] + long_arm[1])


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
    ) -> tuple[tuple[int, int], tuple[int, int], list[int], int]:
        options: list[tuple[tuple[int, int], tuple[int, int], list[int], int]] = []
        for stem in arms:
            witnesses: list[int] = []
            ok = True
            for rr, cc in _tip_lateral_cells(center, stem):
                if not _in_bounds(g.shape, rr, cc) or comp_id[rr, cc] < 0:
                    ok = False
                    break
                witnesses.append(int(comp_id[rr, cc]))
            if not ok:
                continue
            source = witnesses[0]
            if any(comp_colors[w] != comp_colors[source] for w in witnesses[1:]):
                continue
            dest = first_component_from(center, stem, start=2)
            if dest is None or int(dest) == int(source):
                continue
            options.append((stem, _other_arm(arms, stem), witnesses, int(dest)))
        assert len(options) == 1, "L marker must have a unique short-arm assignment"
        return options[0]

    for marker in markers:
        if marker["kind"] in {"normal", "colored"}:
            stem, long_arm, source_parts, dest = resolve_l_assignment(marker["center"], marker["arms"])
            source = source_parts[0]
            for part in source_parts[1:]:
                dsu.union(source, part)
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
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    for k in range(1, int(n) + 1):
        coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == k)}
        if len(coords) != 5:
            continue
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
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
    Repaint regions from L-shaped control markers, erase those markers, and optionally rotate the result using a color-9 L marker.

    Input structure:
    - The grid is partitioned into monochrome regions.
    - Color 1 forms 3-cell orthogonal L markers on boundaries between regions.
    - Some controls use a colored elbow with exactly two singleton color-1 arm cells; that elbow color is the literal replacement color.
    - An optional 5-cell L of color 9 is embedded inside one region and signals a 90-degree rotation of the final output.

    Rule:
    - Each control has two possible arm assignments. The short arm is the one whose forward ray reaches a different base region when you continue one more step beyond the marker.
    - That short arm points to the destination region.
    - The source region is read from the two lateral neighbors of the stem tip at distance 2 from the elbow. A normal L copies that source region's original color; a colored elbow uses its own color instead.
    - Every control cell is repainted with the final color of the source-side region. A color-9 L rotates the finished output: top-right corner means CCW 90 degrees, top-left corner means CW 90 degrees.

    Key insight:
    - You must first decide which L arm is the short arm by following each arm outward, then look sideways around that arm's tip to find the donor color.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (combo) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
MARKER_MIN = 4
MARKER_MAX = 4
NORMAL_MAX = 6
COLORED_MAX = 2
COLOR_POOL = [c for c in range(10) if c not in {1, 9}]
REGION_COUNTS = [5, 6, 7]
REGION_SIZE_FACTOR = 3.0
SEED_ATTEMPTS = 200
GEN_RETRIES = 300
ROTATION_CYCLE = (None, "top_right", None, "top_left")
_GENERATION_COUNTER = 0


def _l_cells(center: tuple[int, int], stem: tuple[int, int], long_arm: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r, c), (r + stem[0], c + stem[1]), (r + long_arm[0], c + long_arm[1])}


def _l_one_cells(center: tuple[int, int], stem: tuple[int, int], long_arm: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + stem[0], c + stem[1]), (r + long_arm[0], c + long_arm[1])}


def _l_neighborhood(center: tuple[int, int], stem: tuple[int, int], long_arm: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _l_cells(center, stem, long_arm)
    out = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                out.add(nb)
    return out


def _destination_shape(stem: tuple[int, int], long_arm: tuple[int, int]) -> set[tuple[int, int]]:
    if stem == (0, 1) and long_arm == (-1, 0):
        return {(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (1, 5)}
    if stem == (0, 1) and long_arm == (1, 0):
        return {(0, 1), (0, 2), (0, 3), (0, 4), (-1, 3), (-1, 4), (-1, 5)}
    if stem == (0, -1) and long_arm == (-1, 0):
        return {(0, -1), (0, -2), (0, -3), (0, -4), (1, -3), (1, -4), (1, -5)}
    if stem == (0, -1) and long_arm == (1, 0):
        return {(0, -1), (0, -2), (0, -3), (0, -4), (-1, -3), (-1, -4), (-1, -5)}
    raise AssertionError("generator only uses horizontal stems")


def _choose_rotation_mode() -> str | None:
    # Cycle rotation presence/orientation so a short ARC training set reliably shows both paths.
    return ROTATION_CYCLE[_GENERATION_COUNTER % len(ROTATION_CYCLE)]


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
        if int(mask.sum()) < MIN_REGION_CELLS:
            return None
        _, n_cc = label_mask(mask, connectivity=4)
        if n_cc != 1:
            return None
    return region


def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    H, W = region.shape
    undirected = set()
    for r in range(H):
        for c in range(W):
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    a = int(region[r, c])
                    b = int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    candidates = []
    for r in range(2, H - 2):
        for c in range(2, W - 2):
            center = (r, c)
            for stem in DIRS:
                for long_arm in _orthogonal_dirs(stem):
                    cells = _l_cells(center, stem, long_arm)
                    if not all(0 <= rr < H and 0 <= cc < W for rr, cc in cells):
                        continue
                    if cells & occupied_cells:
                        continue
                    if _l_neighborhood(center, stem, long_arm) & occupied_cells:
                        continue
                    if cells & occupied_neighbors:
                        continue

                    stem_cell = (r + stem[0], c + stem[1])
                    tip_cell = (r + 2 * stem[0], c + 2 * stem[1])
                    long_cell = (r + long_arm[0], c + long_arm[1])
                    swap_tip = (r + 2 * long_arm[0], c + 2 * long_arm[1])
                    probe_cells = _tip_lateral_cells(center, stem)
                    legacy = _legacy_diagonal_cell(center, stem, long_arm)
                    needed = [center, stem_cell, tip_cell, long_cell, swap_tip, legacy, *probe_cells]
                    if not all(0 <= rr < H and 0 <= cc < W for rr, cc in needed):
                        continue
                    if any((rr, cc) in occupied_cells for rr, cc in [tip_cell, swap_tip, legacy, *probe_cells]):
                        continue

                    if int(region[center]) != src or int(region[long_cell]) != src:
                        continue
                    if int(region[stem_cell]) != dst or int(region[tip_cell]) != dst:
                        continue
                    if any(int(region[rr, cc]) != src for rr, cc in probe_cells):
                        continue
                    if int(region[legacy]) == src:
                        continue
                    if int(region[swap_tip]) != src:
                        continue

                    candidates.append((center, stem, long_arm))
    return candidates


def _build_band_region_map() -> tuple[np.ndarray, list[tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int]]]] | None:
    H = random.randint(18, 26)
    W = random.randint(22, GRID_SIZE_MAX)
    ambient = 2
    top_src = 0
    bot_src = 1
    region = np.full((H, W), ambient, dtype=int)

    top_r = random.randint(5, 6)
    bot_r = random.randint(H - 7, H - 6)
    region[top_r - 2:top_r + 3, :] = top_src
    region[bot_r - 2:bot_r + 3, :] = bot_src

    marker_defs: list[tuple[int, int, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    base_defs = [
        (top_src, 3, (top_r, 4), (0, 1)),
        (top_src, 4, (top_r, W - 5), (0, -1)),
        (bot_src, 5, (bot_r, 4), (0, 1)),
        (bot_src, 6, (bot_r, W - 5), (0, -1)),
    ]
    for src_id, dst_id, center, stem in base_defs:
        long_arm = random.choice(_orthogonal_dirs(stem))
        legacy = _legacy_diagonal_cell(center, stem, long_arm)
        shape = {(center[0] + dr, center[1] + dc) for dr, dc in _destination_shape(stem, long_arm)}
        if not all(0 <= rr < H and 0 <= cc < W for rr, cc in shape | {legacy}):
            return None
        region[legacy] = ambient
        for rr, cc in shape:
            region[rr, cc] = dst_id
        marker_defs.append((src_id, dst_id, center, stem, long_arm))

    for rid in [0, 1, 3, 4, 5, 6]:
        mask = region == rid
        if int(mask.sum()) < MIN_REGION_CELLS:
            return None
        _, n_cc = label_mask(mask, connectivity=4)
        if n_cc != 1:
            return None
    if int((region == ambient).sum()) < MIN_REGION_CELLS:
        return None
    return region, marker_defs


def _register_marker(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    long_arm: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
    cells = _l_cells(center, stem, long_arm)
    if kind == "normal":
        occupied_cells |= cells
        occupied_neighbors |= (_l_neighborhood(center, stem, long_arm) - cells)
        return
    arm_cells = _l_one_cells(center, stem, long_arm)
    occupied_cells |= arm_cells
    occupied_cells.add(center)
    for ar, ac in arm_cells:
        for dr, dc in DIRS:
            nb = (ar + dr, ac + dc)
            if nb != center and nb not in arm_cells:
                occupied_neighbors.add(nb)
    for dr, dc in DIRS:
        nb = (center[0] + dr, center[1] + dc)
        if nb not in arm_cells:
            occupied_neighbors.add(nb)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int], int, int, int | None]],
) -> None:
    for kind, center, stem, long_arm, _src, _dst, target_color in placed_markers:
        if kind == "normal":
            for rr, cc in _l_cells(center, stem, long_arm):
                grid[rr, cc] = 1
        else:
            grid[center[0], center[1]] = int(target_color)
            for rr, cc in _l_one_cells(center, stem, long_arm):
                grid[rr, cc] = 1


def _stamp_markers_output(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int], int, int, int | None]],
    out_colors: list[int],
) -> None:
    for _kind, center, stem, long_arm, src, _dst, _target_color in placed_markers:
        color = int(out_colors[src])
        for rr, cc in _l_cells(center, stem, long_arm):
            grid[rr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int], int, int, int | None]],
    occupied_neighbors: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
    desired_orientation: str | None,
) -> str | None:
    if desired_orientation is None:
        return None
    orient = desired_orientation
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, stem, long_arm, *_rest in placed_markers:
        all_marker_cells |= _l_cells(center, stem, long_arm)

    candidates = []
    for r in range(H):
        for c in range(W):
            cells = [(r + dr, c + dc) for dr, dc in offsets]
            if not all(0 <= rr < H and 0 <= cc < W for rr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[rr, cc]) == rid for rr, cc in cells):
                continue
            if any((rr, cc) in all_marker_cells for rr, cc in cells):
                continue
            if any((rr, cc) in occupied_neighbors for rr, cc in cells):
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
    built = _build_band_region_map()
    if built is None:
        return None
    region, base_markers = built
    n_regions = int(region.max()) + 1
    H, W = region.shape

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    for _src, _dst, center, stem, long_arm in base_markers:
        _register_marker("normal", center, stem, long_arm, occupied_cells, occupied_neighbors)

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    top_markers = [marker for marker in base_markers if marker[0] == 0]
    bot_markers = [marker for marker in base_markers if marker[0] == 1]
    assert len(top_markers) == 2 and len(bot_markers) == 2
    colored_base = [random.choice(top_markers), random.choice(bot_markers)]
    colored_keys = {(center, stem, long_arm) for _src, _dst, center, stem, long_arm in colored_base}
    normal_base = [
        marker
        for marker in base_markers
        if (marker[2], marker[3], marker[4]) not in colored_keys
    ]
    assert len(normal_base) == 2 and len(colored_base) == 2

    out_colors = list(input_colors)
    colored_targets: list[int] = []
    used_target_colors: set[int] = set()
    for _src, dst, _center, _stem, _long_arm in colored_base:
        src = _src
        choices = [
            c
            for c in pool
            if c != input_colors[dst] and c != input_colors[src] and c not in used_target_colors
        ]
        if not choices:
            choices = [c for c in pool if c != input_colors[dst] and c != input_colors[src]]
        if not choices:
            return None
        target_color = random.choice(choices)
        colored_targets.append(target_color)
        used_target_colors.add(target_color)

    for src, dst, _center, _stem, _long_arm in normal_base:
        out_colors[dst] = input_colors[src]
    for (_src, dst, _center, _stem, _long_arm), tc in zip(colored_base, colored_targets):
        out_colors[dst] = tc

    placed_markers = [("normal", center, stem, long_arm, src, dst, None) for src, dst, center, stem, long_arm in normal_base]
    placed_markers += [
        ("colored", center, stem, long_arm, src, dst, tc)
        for (src, dst, center, stem, long_arm), tc in zip(colored_base, colored_targets)
    ]

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    l9_orient = _try_place_l9(
        region,
        placed_markers,
        occupied_neighbors,
        input_grid,
        output_grid,
        out_colors,
        desired_orientation=_choose_rotation_mode(),
    )
    if l9_orient == "top_right":
        output_grid = np.rot90(output_grid, 1)
    elif l9_orient == "top_left":
        output_grid = np.rot90(output_grid, 3)

    try:
        result = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(result, output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    global _GENERATION_COUNTER
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            _GENERATION_COUNTER += 1
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
