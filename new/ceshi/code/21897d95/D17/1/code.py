# Variant of 21897d95
# Dimension: D17 — 颜色1的保留角色 (meta)
# Change: 颜色1（蓝色）专门保留给T标记单元，不用于任何区域着色... → 不同T标记使用不同保留颜色
#
# Design analysis
# spec:
#   variant_core:
#     original: every T control used color 1 for its structural cells, so marker parsing could hard-code "1 means control".
#     variant: T controls use one of three reserved structural colors {1,3,7}. The reserved color is part of the instruction:
#       7-priority controls override 3-priority controls, which override 1-priority controls, whenever multiple T markers point to the same destination region.
#   concrete_encoding:
#     normal_T:
#       shape: a monochrome 4-cell T tetromino, but its color can be 1, 3, or 7.
#       effect: copy the source region's original color into the destination region if this T has the highest priority among markers targeting that region.
#     colored_T:
#       shape: a non-reserved center cell plus three singleton branch cells of one reserved color forming a T.
#       effect: if selected by priority, the destination region changes to the center color.
#     region_palette:
#       constraint: region colors exclude 1, 3, 7 so every occurrence of these colors is visibly marker-only.
#   behavior_trace:
#     original_flow:
#       - detect every color-1 T
#       - each destination region has at most one effective control in generated samples
#       - apply that control directly
#     variant_flow:
#       - detect every T built from one reserved color in {1,3,7}
#       - if two or more Ts point to the same region, compare only their marker colors
#       - keep the recolor instruction from the strongest reserved color and ignore weaker competing instructions for that destination
#     example:
#       - a color-1 T points region A -> region D, and a color-7 T points region B -> the same region D
#       - original code would have rejected or ignored this multi-target situation as conflicting / unsupported by generation
#       - new code resolves it deterministically: region D takes B's original color because the color-7 marker outranks the color-1 marker
#   complexity_calibration:
#     target: enough extra structure to make D17 visible without changing the geometric language of the task
#     choice: only three reserved marker colors and a single fixed priority chain; no per-sample remapping and no color-id arithmetic
#     generator_guard: every sample contains at least one contested destination with at least two different reserved marker colors, so the extra semantics are exercised
#   implicit_assumption_audit:
#     main_old_assumptions:
#       - label_mask(g == 1) is sufficient to find all T branches
#       - all non-1 cells are eligible region / small-marker colors
#       - conflicting controls for one destination should assert
#     main_new_replacements:
#       - scan each reserved marker color separately, then unify their detected markers
#       - exclude the full reserved set from base-region labeling
#       - resolve destination conflicts by priority, only asserting when equal-priority controls disagree
#   placement_legality:
#     - generator still uses the original T geometry and spacing checks
#     - contested destinations reuse the same target region but never reuse the same (src,dst) pair or the same reserved color on one destination
#     - colored-T branch cells stay singleton components because occupied-neighbor buffering is preserved
#   visual_observability:
#     - region colors never use 1,3,7, so those colors are always read as symbolic ink
#     - at least one region per sample is targeted by multiple differently colored Ts, making the priority rule visible in the output
#     - output still erases all markers into source-side colors, so the only way to predict the winning recolor is to decode marker color priority
#   cleanliness:
#     - keep optional L-shaped color-9 rotation marker unchanged
#     - keep small scrap absorption unchanged
#     - avoid adding new object types beyond multi-color Ts

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
RESERVED_T_COLORS = (1, 3, 7)
T_PRIORITY = {1: 0, 3: 1, 7: 2}


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


def _label_base_components(
    g: np.ndarray,
    control: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
    comp_areas: list[int] = []
    cid = 0
    blocked = set(RESERVED_T_COLORS)
    for col in sorted(set(map(int, g.ravel())) - blocked):
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
    reserved_labels: dict[int, np.ndarray] = {}
    reserved_components: dict[int, dict[int, set[tuple[int, int]]]] = {}
    reserved_sizes: dict[int, dict[int, int]] = {}
    markers: list[dict] = []
    used_reserved: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)

    for marker_color in RESERVED_T_COLORS:
        labels, n_labels = label_mask(g == marker_color, connectivity=4)
        reserved_labels[marker_color] = labels
        comps = {
            k: {tuple(map(int, xy)) for xy in np.argwhere(labels == k)}
            for k in range(1, int(n_labels) + 1)
        }
        reserved_components[marker_color] = comps
        reserved_sizes[marker_color] = {k: len(cells) for k, cells in comps.items()}
        for cells in comps.values():
            parsed = _parse_tetromino(cells)
            if parsed is None:
                continue
            center, stem, bar = parsed
            markers.append(
                {
                    "kind": "normal",
                    "center": center,
                    "stem": stem,
                    "bar": bar,
                    "cells": cells,
                    "marker_color": marker_color,
                }
            )
            used_reserved |= cells
            for r, c in cells:
                control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) in RESERVED_T_COLORS:
                continue
            adjacent = []
            for d in DIRS:
                rr, cc = r + d[0], c + d[1]
                if not _in_bounds(g.shape, rr, cc):
                    continue
                marker_color = int(g[rr, cc])
                if marker_color not in RESERVED_T_COLORS:
                    continue
                label = int(reserved_labels[marker_color][rr, cc])
                adjacent.append((d, (rr, cc), marker_color, label))
            if len(adjacent) != 3:
                continue
            marker_colors = {marker_color for _d, _cell, marker_color, _label in adjacent}
            if len(marker_colors) != 1:
                continue
            marker_color = next(iter(marker_colors))
            branches: list[tuple[int, int]] = []
            cells: list[tuple[int, int]] = []
            ok = True
            for d, cell, _color, label in adjacent:
                if label == 0 or reserved_sizes[marker_color][label] != 1 or cell in used_reserved:
                    ok = False
                    break
                branches.append(d)
                cells.append(cell)
            if not ok:
                continue
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
                    "marker_color": marker_color,
                }
            )
            used_reserved |= set(cells)
            for rr, cc in marker_cells:
                control[rr, cc] = True

    all_reserved_cells = {
        tuple(map(int, xy))
        for xy in np.argwhere(np.isin(g, np.array(RESERVED_T_COLORS, dtype=int)))
    }
    assert all_reserved_cells == used_reserved, "all reserved T-color pixels must belong to a T control marker"

    for col in sorted(set(map(int, g.ravel())) - set(RESERVED_T_COLORS)):
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

    def first_component_from(
        center: tuple[int, int],
        direction: tuple[int, int],
        start: int,
    ) -> int | None:
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
            dest = first_component_from(center, stem, start=2)
            assert dest is not None, "T marker must point at a destination region"

            source_parts: list[int] = []
            for bar_dir in marker["bar"]:
                rr = center[0] + stem[0] + bar_dir[0]
                cc = center[1] + stem[1] + bar_dir[1]
                assert _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0
                source_parts.append(int(comp_id[rr, cc]))
            source = source_parts[0]
            for part in source_parts[1:]:
                assert comp_colors[part] == comp_colors[source], "T source diagonals must share a color"
                dsu.union(source, part)

            for comp in adjacent_component_counts(marker["cells"]):
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
            marker["dest_comp"] = dest
            marker["source_comp"] = source
            marker["priority"] = T_PRIORITY[int(marker["marker_color"])]
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
    best_control: dict[int, tuple[int, int]] = {}
    for marker in markers:
        if marker["kind"] not in {"normal", "colored"}:
            marker["source"] = dsu.find(marker["source_comp"])
            continue
        dest = dsu.find(marker["dest_comp"])
        source = dsu.find(marker["source_comp"])
        marker["source"] = source
        replacement = (
            int(marker["target_color"])
            if marker["kind"] == "colored"
            else int(root_color[source])
        )
        prev = best_control.get(dest)
        if prev is None or marker["priority"] > prev[0]:
            best_control[dest] = (int(marker["priority"]), replacement)
        elif marker["priority"] == prev[0]:
            assert prev[1] == replacement, "equal-priority T controls for one region must agree"

    for dest, (_priority, replacement) in best_control.items():
        final_color[dest] = int(replacement)

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
    Decode multi-color T controls, repaint the regions they govern, and optionally rotate via a color-9 L marker.

    Input structure:
    - The grid is partitioned into single-color regions whose colors avoid the reserved T-marker palette {1, 3, 7}.
    - A normal control is a 4-cell T tetromino drawn entirely in one reserved marker color.
    - A colored control has a non-reserved center cell with three singleton neighbors of one reserved marker color forming a T.
    - An optional 5-cell L marker of color 9 can sit inside a region and request a 90-degree output rotation.

    Rule:
    - The T stem points toward the destination region, while the two diagonal cells behind the stem identify the source side.
    - A normal T wants the destination to copy the source region's original color; a colored-center T wants the destination to take the center color.
    - If several T markers point to the same destination, only the strongest reserved color applies: color 7 overrides color 3, and color 3 overrides color 1.
    - Every marker cell is replaced by the final color of the source-side region, and the L marker rotates the completed output: top-right corner means CCW, top-left means CW.

    Key insight:
    - First separate structural marker colors from region colors, then read each T geometrically as before, but resolve competing instructions by the marker color's fixed priority.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (priority_multicolor_t) ---------------

# ── Tunable constants ──────────────────────────────────────────────────
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6]
REGION_WEIGHTS = [3, 3, 2]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

# Region colors avoid reserved T colors and the L9 marker color.
COLOR_POOL = [0, 2, 4, 5, 6, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 300

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _choose_marker_counts(max_pairs: int) -> tuple[int, int, int] | None:
    """Return (n_markers, n_normal, n_colored) or None if infeasible."""
    max_possible = min(MARKER_MAX, max_pairs)
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


def _build_marker_plan(
    directed_pairs: list[tuple[int, int]],
    input_colors: list[int],
    n_markers: int,
    n_normal: int,
    n_colored: int,
) -> list[dict] | None:
    incoming: dict[int, list[int]] = defaultdict(list)
    for src, dst in directed_pairs:
        incoming[int(dst)].append(int(src))
    for dst in list(incoming):
        incoming[dst] = sorted(set(incoming[dst]))

    contested = [dst for dst, srcs in incoming.items() if len(srcs) >= 2]
    if not contested:
        return None

    defs: list[dict] = []
    used_pairs: set[tuple[int, int]] = set()
    used_marker_colors: dict[int, set[int]] = defaultdict(set)

    def add_entry(src: int, dst: int, marker_color: int) -> None:
        defs.append(
            {
                "src": int(src),
                "dst": int(dst),
                "marker_color": int(marker_color),
            }
        )
        used_pairs.add((int(src), int(dst)))
        used_marker_colors[int(dst)].add(int(marker_color))

    heavy_contested = [dst for dst in contested if len(incoming[dst]) >= 3]
    first_dst_pool = heavy_contested if heavy_contested and n_markers >= 5 else contested
    first_dst = random.choice(first_dst_pool)
    max_group = min(3, len(incoming[first_dst]), len(RESERVED_T_COLORS), n_markers)
    group_size = 3 if max_group >= 3 and n_markers >= 5 and random.random() < 0.7 else 2
    srcs = random.sample(incoming[first_dst], group_size)
    marker_colors = random.sample(list(RESERVED_T_COLORS), group_size)
    for src, marker_color in zip(srcs, marker_colors):
        add_entry(src, first_dst, marker_color)

    other_dests = [dst for dst in incoming if dst != first_dst]
    random.shuffle(other_dests)
    for dst in other_dests:
        if len(defs) >= n_markers:
            break
        choices = [src for src in incoming[dst] if (src, dst) not in used_pairs]
        if not choices:
            continue
        add_entry(random.choice(choices), dst, random.choice(list(RESERVED_T_COLORS)))

    expand_dests = list(incoming.keys())
    random.shuffle(expand_dests)
    progress = True
    while len(defs) < n_markers and progress:
        progress = False
        for dst in expand_dests:
            if len(defs) >= n_markers:
                break
            choices = [src for src in incoming[dst] if (src, dst) not in used_pairs]
            colors = [c for c in RESERVED_T_COLORS if c not in used_marker_colors[dst]]
            if not choices or not colors:
                continue
            add_entry(random.choice(choices), dst, random.choice(colors))
            progress = True

    if len(defs) < n_markers:
        return None

    random.shuffle(defs)
    colored_indices = set(random.sample(range(len(defs)), n_colored))
    for i, entry in enumerate(defs):
        if i in colored_indices:
            forbidden = {input_colors[entry["dst"]]}
            alt_forbidden = forbidden | {input_colors[entry["src"]]}
            candidates = [c for c in COLOR_POOL if c not in alt_forbidden]
            if not candidates:
                candidates = [c for c in COLOR_POOL if c not in forbidden]
            if not candidates:
                return None
            entry["kind"] = "colored"
            entry["target_color"] = random.choice(candidates)
        else:
            entry["kind"] = "normal"
            entry["target_color"] = None

    assert sum(1 for entry in defs if entry["kind"] == "normal") == n_normal
    assert sum(1 for entry in defs if entry["kind"] == "colored") == n_colored
    return defs


def _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W):
    """Find all valid (center, stem) for placing a T marker from src toward dst."""
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in _DIRS:
                bar = _get_bar(stem)
                cells = _t_cells((r, c), stem)
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                if cells & occupied_cells:
                    continue
                if _t_neighborhood((r, c), stem) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue
                if not all(
                    0 <= r + stem[0] + bd[0] < H
                    and 0 <= c + stem[1] + bd[1] < W
                    and region[r + stem[0] + bd[0], c + stem[1] + bd[1]] == src
                    for bd in bar
                ):
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
                    candidates.append(((r, c), stem))
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
    for kind, center, stem, _src, _dst, tc, marker_color in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = marker_color
        else:
            grid[center[0], center[1]] = tc
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = marker_color


def _stamp_markers_output(grid, placed_markers, out_colors):
    """Fill marker cells in the output grid with the source region's final color."""
    for _kind, center, stem, src, _dst, _tc, _marker_color in placed_markers:
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
    for _kind, center, stem, *_rest in placed_markers:
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
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    directed_pairs = _find_adjacent_pairs(region)
    counts = _choose_marker_counts(len(directed_pairs))
    if counts is None:
        return None
    n_markers, n_normal, n_colored = counts

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    marker_defs = _build_marker_plan(directed_pairs, input_colors, n_markers, n_normal, n_colored)
    if marker_defs is None:
        return None

    out_colors = list(input_colors)
    best_control: dict[int, tuple[int, int]] = {}
    for entry in marker_defs:
        replacement = (
            entry["target_color"]
            if entry["kind"] == "colored"
            else input_colors[entry["src"]]
        )
        priority = T_PRIORITY[entry["marker_color"]]
        prev = best_control.get(entry["dst"])
        if prev is None or priority > prev[0]:
            best_control[entry["dst"]] = (priority, int(replacement))
        elif priority == prev[0] and prev[1] != int(replacement):
            return None
    for dst, (_priority, replacement) in best_control.items():
        out_colors[dst] = int(replacement)

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers = []
    random.shuffle(marker_defs)
    for entry in marker_defs:
        cands = _find_valid_placements(
            region,
            entry["src"],
            entry["dst"],
            occupied_cells,
            occupied_neighbors,
            H,
            W,
        )
        if not cands:
            return None
        center, stem = random.choice(cands)
        _register_marker(entry["kind"], center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append(
            (
                entry["kind"],
                center,
                stem,
                entry["src"],
                entry["dst"],
                entry["target_color"],
                entry["marker_color"],
            )
        )

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
    )
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
