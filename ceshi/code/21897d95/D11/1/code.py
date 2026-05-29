# Variant of 21897d95
# Dimension: D11 — 区域颜色数量与约束 (meta)
# Change: 3-6种不同的非颜色1颜色用于区域着色；颜色1保留给T标记... → 区域颜色可重复使用

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D11 variant
# -----------------------------------
# variant_spec:
#   core_change:
#     original: every visible base region had its own non-1 color, so solver code
#       could sometimes treat "same color touching the same marker" as evidence that
#       those pieces belonged to one logical region.
#     variant: the same non-1 color may appear in multiple disconnected visible
#       regions. Region identity must therefore be recovered from 4-connectivity,
#       not color id alone.
#   concrete_pixel_behavior:
#     - The input is still a mosaic of monochrome 4-connected regions, plus T markers
#       in color 1 and an optional 5-cell L marker in color 9.
#     - The new visual difference is that two or more disconnected regions may share
#       the same fill color while remaining distinct because they are not 4-connected.
#     - Adjacent regions are still forced to have different colors; otherwise they
#       would visually merge into a single 4-connected region and the change would not
#       be observable.
#   flow_trace:
#     original_logic:
#       - After removing control markers, the solver labels monochrome connected
#         components.
#       - Around a marker it then unions any adjacent components with the same color.
#       - This is harmless only when every real region color is globally unique.
#     variant_logic:
#       - After removing control markers, each connected monochrome component is taken
#         as a region directly.
#       - A T marker reads its source from the two diagonal cells behind the stem and
#         its destination from the first visible component along the stem ray.
#       - No color-based merging is allowed.
#     example:
#       - Suppose two disconnected red regions exist. A T marker copies the lower red
#         region into a yellow destination while another red island sits elsewhere.
#       - The old color-union heuristic could accidentally merge the wrong red island
#         if both touched marker-adjacent cells of matching color.
#       - The new solver only trusts connectivity, so the remote red island is ignored.
#   complexity_calibration:
#     - Repeated colors are guaranteed by the generator via graph coloring with fewer
#       colors than regions.
#     - At least one repeated color is used by a marker source or destination, so the
#       D11 change is not merely incidental background decoration.
#     - Adjacent same-color regions are forbidden; repeated colors only appear on
#       separated components, which keeps the pattern legible.
#   implicit_assumption_audit:
#     - _label_base_components still works because it already splits by 4-connectivity.
#     - The old unions inside _solve_unrotated relied on global color uniqueness and
#       must be removed.
#     - The new solver assumes that removing marker cells does not split any region
#       into multiple surviving components. The generator now enforces this when it
#       accepts a marker placement.
#     - Small non-rectangular <=5-cell components are still treated as absorbed
#       markers. The generator now deliberately uses one such scrap to create a
#       repeated-color ambiguity that the old color-union heuristic would mishandle.
#   placement_legality:
#     - T markers still cannot overlap, go out of bounds, or touch buffered singleton
#       cells incorrectly.
#     - Every candidate placement is filtered so all affected regions remain
#       4-connected after all currently placed control cells are removed.
#     - L9 is only placed inside a region whose input color is not 9, so the marker is
#       visible and is still uniquely recoverable as a small non-rectangular component.
#   observability_and_cleanliness:
#     - The repeated-color effect is visible because disconnected same-color islands
#       appear in the input itself.
#     - Marker rules remain identical, so the only conceptual jump for the solver is
#       that same color no longer implies same region.
#     - Normal T markers always point across a true region boundary, hence their
#       recoloring effect is still easy to spot.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
    cells: set[tuple[int, int]]
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
    g: np.ndarray, control: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
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


def _collect_markers(
    g: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
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
        markers.append(
            {"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells}
        )
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

    return markers, control


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
    markers, control = _collect_markers(g)
    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"

    def first_component_from(
        center: tuple[int, int], direction: tuple[int, int], start: int
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
            assert all(part == source for part in source_parts[1:]), (
                "generator must keep the source region connected after removing markers"
            )

            marker["dest_comp"] = dest
            marker["source_comp"] = source
        else:
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "small marker must touch its owner region"
            marker["source_comp"] = max(counts, key=lambda c: (counts[c], comp_areas[c]))

    final_color = {cid: int(color) for cid, color in enumerate(comp_colors)}
    for marker in markers:
        if marker["kind"] not in {"normal", "colored"}:
            marker["source"] = int(marker["source_comp"])
            continue
        dest = int(marker["dest_comp"])
        source = int(marker["source_comp"])
        marker["source"] = source
        replacement = (
            int(marker["target_color"])
            if marker["kind"] == "colored"
            else int(comp_colors[source])
        )
        assert final_color[dest] in {int(comp_colors[dest]), replacement}, (
            "conflicting T controls for one region"
        )
        final_color[dest] = replacement

    out = np.full(g.shape, -1, dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[cid])

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
                raise AssertionError(f"unobserved L9 corner at ({cr},{cc})")
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor regions from boundary T markers, erase the controls, and optionally rotate via a color-9 L marker.

    Input structure:
    - The grid is partitioned into monochrome 4-connected regions, but different disconnected regions may reuse the same non-1 color.
    - Color 1 still marks T-shaped controls on boundaries, and some controls use a colored center with three singleton color-1 branches.
    - A tiny non-rectangular non-1 scrap may appear and is absorbed into one adjacent region; an optional 5-cell L of color 9 acts as a rotation control.

    Rule:
    - For any T marker, the stem indicates the destination region and the two diagonal cells behind that stem identify the source region.
    - A plain color-1 T copies the source region's original color into the destination; a colored-center T uses its center color instead.
    - After recoloring, every control cell is painted with the final color of the source-side region, and any tiny scrap or L9 marker is absorbed into its owner region.
    - If the L marker is ┐ the finished grid rotates 90 degrees counterclockwise; if it is ┌ the finished grid rotates 90 degrees clockwise.

    Key insight:
    - Matching colors are no longer enough to identify a region: the decisive cue is which 4-connected component the marker geometry touches.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (l9rot + repeated region colors) ---------------

# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [2, 3, 3, 1, 1]

# Marker counts
MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

# Region colors: reuse allowed, but 1 remains reserved for T markers
COLOR_POOL = [c for c in range(10) if c != 1]
DISTINCT_COLOR_MIN = 3
DISTINCT_COLOR_MAX = 6

# L9 rotation marker
L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

# Retry budget
SEED_ATTEMPTS = 200
GEN_RETRIES = 400

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bar(stem):
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center, stem):
    r, c = center
    bar = _get_bar(stem)
    return {
        (r, c),
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_branch_cells(center, stem):
    r, c = center
    bar = _get_bar(stem)
    return {
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_neighborhood(center, stem):
    cells = _t_cells(center, stem)
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


def _adjacency_map(region: np.ndarray) -> dict[int, set[int]]:
    n_regions = int(region.max()) + 1
    adj = {rid: set() for rid in range(n_regions)}
    for a, b in _find_adjacent_pairs(region):
        adj[a].add(b)
    return adj


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


def _region_remains_connected(
    region: np.ndarray, rid: int, removed_cells: set[tuple[int, int]]
) -> bool:
    mask = region == int(rid)
    if not mask.any():
        return False
    remain = mask.copy()
    for r, c in removed_cells:
        if 0 <= r < region.shape[0] and 0 <= c < region.shape[1] and int(region[r, c]) == int(rid):
            remain[r, c] = False
    if not remain.any():
        return False
    _, n = label_mask(remain, connectivity=4)
    return int(n) == 1


def _placement_keeps_regions_connected(
    region: np.ndarray, cells: set[tuple[int, int]], occupied_cells: set[tuple[int, int]]
) -> bool:
    trial_removed = occupied_cells | cells
    affected = {int(region[r, c]) for r, c in cells}
    return all(_region_remains_connected(region, rid, trial_removed) for rid in affected)


def _find_valid_placements(
    region,
    src,
    dst,
    occupied_cells,
    occupied_neighbors,
    H,
    W,
    *,
    require_connected: bool = True,
):
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
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == int(src)
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
                    found = int(region[nr, nc]) == int(dst)
                    break
                if not found:
                    continue
                if require_connected and not _placement_keeps_regions_connected(region, cells, occupied_cells):
                    continue
                candidates.append(((r, c), stem))
    return candidates


def _register_marker(kind, center, stem, occupied_cells, occupied_neighbors):
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
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid, placed_markers):
    for kind, center, stem, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = tc
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(grid, placed_markers, out_colors):
    for _kind, center, stem, src, _dst, _tc in placed_markers:
        color = out_colors[src]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors):
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
            if input_grid[r, c] == 9:
                continue
            candidates.append((r, c, rid))

    if not candidates:
        return None
    lr, lc, rid = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[lr + dr, lc + dc] = 9
        output_grid[lr + dr, lc + dc] = out_colors[rid]
    return orient


def _backtrack_region_coloring(
    order: list[int],
    adj: dict[int, set[int]],
    palette: list[int],
    idx: int,
    assignment: dict[int, int],
) -> dict[int, int] | None:
    if idx == len(order):
        return assignment.copy() if len(set(assignment.values())) == len(palette) else None
    rid = order[idx]
    used_neighbors = {assignment[nbr] for nbr in adj[rid] if nbr in assignment}
    colors = palette[:]
    random.shuffle(colors)
    for color in colors:
        if color in used_neighbors:
            continue
        assignment[rid] = color
        result = _backtrack_region_coloring(order, adj, palette, idx + 1, assignment)
        if result is not None:
            return result
        del assignment[rid]
    return None


def _same_color_groups(
    n_regions: int, same_pairs: list[tuple[int, int]]
) -> tuple[dict[int, list[int]], dict[int, int]]:
    parent = list(range(n_regions))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in same_pairs:
        union(int(a), int(b))

    groups: dict[int, list[int]] = {}
    root_of: dict[int, int] = {}
    for rid in range(n_regions):
        root = find(rid)
        root_of[rid] = root
        groups.setdefault(root, []).append(rid)
    return groups, root_of


def _assign_reused_region_colors(
    region: np.ndarray, same_pairs: list[tuple[int, int]]
) -> list[int] | None:
    adj = _adjacency_map(region)
    n_regions = len(adj)
    groups, root_of = _same_color_groups(n_regions, same_pairs)

    for rid, neighbors in adj.items():
        if any(root_of[rid] == root_of[nbr] for nbr in neighbors):
            return None

    group_adj = {root: set() for root in groups}
    for rid, neighbors in adj.items():
        rr = root_of[rid]
        for nbr in neighbors:
            rn = root_of[nbr]
            if rr != rn:
                group_adj[rr].add(rn)

    max_distinct = min(DISTINCT_COLOR_MAX, len(group_adj))
    if max_distinct < DISTINCT_COLOR_MIN:
        return None
    order = sorted(group_adj, key=lambda rid: (-len(group_adj[rid]), rid))
    for n_distinct in random.sample(
        list(range(DISTINCT_COLOR_MIN, max_distinct + 1)),
        k=max_distinct - DISTINCT_COLOR_MIN + 1,
    ):
        for _ in range(6):
            palette = random.sample(COLOR_POOL, n_distinct)
            result = _backtrack_region_coloring(order, group_adj, palette, 0, {})
            if result is None:
                continue
            return [result[root_of[rid]] for rid in range(n_regions)]
    return None


def _adjacent_region_ids_for_cells(region, cells):
    touched = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < region.shape[0] and 0 <= nc < region.shape[1] and (nr, nc) not in cells:
                touched.add(int(region[nr, nc]))
    return touched


SCRAP_SHAPES = [
    {(0, 0), (1, 0), (0, 1)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 1), (1, 0), (1, 1)},
]
SCRAP_COLOR_POOL = [c for c in COLOR_POOL if c != 9]


def _shape_neighborhood(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                out.add(nb)
    return out


def _find_scrap_witness(region, directed_pairs):
    adj = _adjacency_map(region)
    areas = {rid: int((region == rid).sum()) for rid in adj}
    incoming: dict[int, list[int]] = {}
    for src, dst in directed_pairs:
        incoming.setdefault(dst, []).append(src)

    H, W = region.shape
    candidates = []
    for r in range(H - 1):
        for c in range(W - 1):
            for shape in SCRAP_SHAPES:
                cells = {(r + dr, c + dc) for dr, dc in shape}
                if not _placement_keeps_regions_connected(region, cells, set()):
                    continue
                counts: Counter = Counter()
                for rr, cc in cells:
                    for dr, dc in _DIRS:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cells:
                            counts[int(region[nr, nc])] += 1
                if len(counts) < 2:
                    continue
                owner = max(counts, key=lambda rid: (counts[rid], areas[rid]))
                for decoy in counts:
                    if decoy == owner or decoy in adj[owner]:
                        continue
                    src_b_options = [
                        src_b for src_b in incoming.get(decoy, []) if src_b != owner
                    ]
                    if src_b_options:
                        candidates.append(
                            (owner, decoy, cells, random.choice(src_b_options))
                        )
    return random.choice(candidates) if candidates else None


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
    if n_normal < 1:
        return None

    witness = _find_scrap_witness(region, directed_pairs)
    if witness is None:
        return None
    owner_a, decoy_b, scrap_cells, src_b = witness

    reserved_normal = [(src_b, decoy_b)]
    reserved_set = set(reserved_normal)
    random.shuffle(directed_pairs)
    extra_pairs = []
    dest_used = {owner_a, decoy_b}
    for src, dst in directed_pairs:
        if (src, dst) in reserved_set or dst in dest_used:
            continue
        extra_pairs.append((src, dst))
        dest_used.add(dst)
        if len(extra_pairs) >= MARKER_MAX - 2:
            break

    max_total = min(n_markers, 2 + len(extra_pairs))
    if max_total < MARKER_MIN:
        return None
    n_markers = max_total
    n_colored = min(n_colored, n_markers - 2)
    n_normal = n_markers - n_colored

    random.shuffle(extra_pairs)
    selected_extra = extra_pairs[: n_markers - 2]
    normal_pairs = reserved_normal + selected_extra[: n_normal - 1]
    colored_pairs = selected_extra[n_normal - 1 :]

    input_colors = _assign_reused_region_colors(region, [(owner_a, decoy_b)])
    if input_colors is None:
        return None

    out_colors = list(input_colors)

    colored_targets = []
    for _src, dst in colored_pairs:
        cands = [c for c in COLOR_POOL if c != input_colors[dst]]
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

    occupied_cells: set[tuple[int, int]] = set(scrap_cells)
    occupied_neighbors: set[tuple[int, int]] = _shape_neighborhood(scrap_cells) - scrap_cells
    placed_markers = []
    for kind, src, dst, tc in all_defs:
        cands = _find_valid_placements(
            region, src, dst, occupied_cells, occupied_neighbors, H, W
        )
        if not cands:
            return None
        center, stem = random.choice(cands)
        _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, tc))

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    scrap_cands = [c for c in SCRAP_COLOR_POOL if c != input_colors[owner_a]]
    if not scrap_cands:
        return None
    scrap_color = random.choice(scrap_cands)
    for r, c in scrap_cells:
        input_grid[r, c] = scrap_color
        output_grid[r, c] = out_colors[owner_a]

    l9_orient = _try_place_l9(
        region, placed_markers, occupied_neighbors | scrap_cells, input_grid, output_grid, out_colors
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
