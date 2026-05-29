# Combo variant of 21897d95 (index 24)
# combo_spec: D8/2+D17/0
# Name: 双色系统分离

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The original task used color 1 as the dedicated T-marker color, so any color-1 cell was guaranteed to be
#      control structure. The rotation marker used color 9 and encoded its direction through L orientation.
#    - In this combo, the recolor-control system moves entirely to color 0. Normal T markers are 4 black cells,
#      and colored-center T markers are "one colored center + three black singleton branches". Color 1 is released
#      back into the ordinary region palette, so a region can now genuinely be color 1.
#    - The rotation system is separated from the T system by both color and role: the L marker always uses one
#      fixed "top-right corner" geometry, while color 9 means CCW and color 8 means CW.
#    - Small trace: if two inputs have the same black T markers and the same fixed L shape, but one L is 9 and the
#      other is 8, the recoloring phase is identical while the final rotation flips from CCW to CW. Separately, a
#      region painted color 1 is no longer a marker at all because only black cells participate in T decoding.
# b) Complexity calibration
#    - The change is visible without exploding the case count: T detection becomes "find black Ts", rotation
#      detection becomes "find one fixed L in {8,9}".
#    - Only two rotation colors are used, and only one T-marker color is used, so the solver still reasons over a
#      small discrete set of symbol types.
#    - To keep the variant legible, ordinary regions are restricted to colors 1..7. That makes the three systems
#      visually distinct: black T controls, bright 8/9 rotation marker, and non-black/non-8/9 regions.
# c) Hidden assumptions review
#    - Every place that previously assumed "all 1s are T cells" must switch to "all 0s are T cells".
#    - The colored-center detector previously skipped color 1 centers; it must now skip color 0 centers instead.
#    - `_label_base_components` and the small-scrap pass must treat color 1 as a legitimate region color.
#    - Rotation decoding can no longer depend on L orientation; it must decode the fixed L shape first and then map
#      color 9 -> CCW, color 8 -> CW.
#    - New assumption introduced by the combo: no ordinary region or colored-center literal uses 8 or 9, otherwise a
#      random 8/9 patch could masquerade as the rotation symbol. The generator enforces that separation.
# d) Placement legality
#    - T markers still use the original non-overlap and neighbor-buffer constraints so singleton black branches stay
#      singleton and full black Ts stay isolated.
#    - The fixed L marker must fit inside one region, avoid all T markers, and avoid their exclusion buffer.
#    - Because regions now may use color 1, the generator explicitly guarantees at least one true region color-1
#      component survives in the input so the D17 change is actually observable.
# e) Visual observability
#    - A human can now see two symbol vocabularies: black T controls and an 8/9 L marker. The black-vs-colored split
#      makes the two instruction channels easier to separate than in the original.
#    - The generator rejects rotation placements whose result is unchanged after rotation, and rejects color-8 cases
#      where CW and CCW would coincidentally look identical, so the color-coded rotation remains learnable.
# f) Visual cleanliness
#    - No region ever uses black, 8, or 9, so markers do not visually blend into ordinary areas.
#    - The L marker stays compact inside one region and never touches T markers, preserving object boundaries.
# g) Change inferability
#    - Color 1 is not used as an invisible sorting key; it is directly visible as an ordinary region color.
#    - Colors 8 and 9 have a local, structural meaning attached to the same L shape, so a few examples suffice to
#      reveal the mapping.
# h) Post-change `main()` assumptions checked against generation
#    - All black cells belong to T markers and no base region uses black.
#    - At most one fixed-shape L marker exists, and if it exists its color is either 8 or 9.
#    - No colored-center T uses 8 or 9, and no base region uses 8 or 9.
#    - If a rotation marker is present, the rotated result is visually different from the unrotated one.

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
T_MARKER_COLOR = 0
ROTATION_BY_COLOR = {9: 1, 8: 3}
ROTATION_L_OFFSETS = [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)]


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


def _label_base_components(g: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
    comp_areas: list[int] = []
    cid = 0
    for col in sorted(set(map(int, g.ravel())) - {T_MARKER_COLOR}):
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
    labels_0, n_0 = label_mask(g == T_MARKER_COLOR, connectivity=4)
    zero_components = {
        k: {tuple(map(int, xy)) for xy in np.argwhere(labels_0 == k)}
        for k in range(1, int(n_0) + 1)
    }
    zero_sizes = {k: len(cells) for k, cells in zero_components.items()}

    markers: list[dict] = []
    used_zeros: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)

    for cells in zero_components.values():
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        markers.append({"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells})
        used_zeros |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == T_MARKER_COLOR:
                continue
            branches: list[tuple[int, int]] = []
            cells: list[tuple[int, int]] = []
            ok = True
            for d in DIRS:
                rr, cc = r + d[0], c + d[1]
                if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != T_MARKER_COLOR:
                    continue
                label = int(labels_0[rr, cc])
                if zero_sizes[label] == 1 and (rr, cc) not in used_zeros:
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
                used_zeros |= set(cells)
                for rr, cc in marker_cells:
                    control[rr, cc] = True

    all_zero_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == T_MARKER_COLOR)}
    assert all_zero_cells == used_zeros, "all black pixels must belong to a T control marker"

    for col in sorted(set(map(int, g.ravel())) - {T_MARKER_COLOR}):
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


def _is_fixed_rotation_l(coords: set[tuple[int, int]]) -> bool:
    if len(coords) != 5:
        return False
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return False
    expected = {(r0 + dr, c1 + dc) for dr, dc in ROTATION_L_OFFSETS}
    return coords == expected


def _find_rotation_instruction(g: np.ndarray) -> int | None:
    found: int | None = None
    for marker_color, rot_k in ROTATION_BY_COLOR.items():
        mask = g == marker_color
        if not mask.any():
            continue
        labels, n = label_mask(mask, connectivity=4)
        for k in range(1, int(n) + 1):
            coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == k)}
            if not _is_fixed_rotation_l(coords):
                continue
            assert found is None, "multiple rotation markers are not supported"
            found = rot_k
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint regions from black T-shaped controls, erase those controls, and optionally rotate using a fixed-shape 8/9 L marker.

    Input structure:
    - The board is partitioned into monochrome regions, and color 1 may now be a normal region color.
    - Black cells (color 0) form the T control system on region boundaries.
    - A colored-center T consists of one literal replacement-color center with three black singleton neighbors.
    - An optional 5-cell L marker always uses the same top-right-corner shape and is colored either 9 or 8.

    Rule:
    - For a black 4-cell T, the unique stem direction points toward the destination region to repaint.
    - The two diagonal cells behind that stem identify the source-side region; a normal T copies that region's original color.
    - A colored-center T keeps the same geometry but uses its center color as the destination replacement instead.
    - Every T-marker cell is filled with the final color of the source-side region.
    - After recoloring, rotate the whole result if the fixed L marker is present: color 9 means CCW 90 degrees, color 8 means CW 90 degrees.

    Key insight:
    - The puzzle now uses two separated symbol channels: black geometry encodes recoloring, while the colored fixed L uses only its color to choose rotation.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    rot_k = _find_rotation_instruction(g)
    return out if rot_k is None else np.rot90(out, rot_k)


# --------------- generator (split black-T / colored-L system) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7]
REGION_WEIGHTS = [3, 3, 2, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

REGION_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7]
ROTATION_MODE_SCHEDULE = [9, 8, None]
COLORED_CENTER_SCHEDULE = [True, False]

SEED_ATTEMPTS = 200
GEN_RETRIES = 300

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GENERATE_CALL_COUNT = 0


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    bar = _get_bar(stem)
    return {
        (r, c),
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_branch_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    bar = _get_bar(stem)
    return {
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_neighborhood(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
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


def _choose_marker_counts(n_regions: int, include_colored_center: bool) -> tuple[int, int, int] | None:
    max_possible = min(MARKER_MAX, n_regions, NORMAL_MAX + (COLORED_MAX if include_colored_center else 0))
    if max_possible < MARKER_MIN:
        return None
    n_markers = random.randint(MARKER_MIN, max_possible)
    if not include_colored_center:
        n_colored = 0
    else:
        min_colored = max(1, n_markers - NORMAL_MAX)
        max_colored = min(COLORED_MAX, n_markers - 2)
        if min_colored > max_colored:
            return None
        n_colored = random.randint(min_colored, max_colored)
    n_normal = n_markers - n_colored
    assert n_normal <= NORMAL_MAX
    return n_markers, n_normal, n_colored


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
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


def _register_marker(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
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


def _choose_input_colors(n_regions: int) -> list[int] | None:
    if n_regions > len(REGION_COLOR_POOL):
        return None
    remaining = [c for c in REGION_COLOR_POOL if c != 1]
    random.shuffle(remaining)
    colors = [1] + remaining[: n_regions - 1]
    random.shuffle(colors)
    return colors


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[tuple]) -> None:
    for kind, center, stem, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = T_MARKER_COLOR
        else:
            grid[center[0], center[1]] = tc
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = T_MARKER_COLOR


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[tuple], out_colors: list[int]) -> None:
    for _kind, center, stem, src, _dst, _tc in placed_markers:
        color = out_colors[src]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _try_place_rotation_marker(
    region: np.ndarray,
    placed_markers: list[tuple],
    occupied_neighbors: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
    desired_rot_color: int | None,
) -> int | None:
    if desired_rot_color is None:
        return None
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, stem, *_rest in placed_markers:
        all_marker_cells |= _t_cells(center, stem)

    candidates = []
    for r in range(H):
        for c in range(W):
            cells = [(r + dr, c + dc) for dr, dc in ROTATION_L_OFFSETS]
            if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if any((cr, cc) in all_marker_cells for cr, cc in cells):
                continue
            if any((cr, cc) in occupied_neighbors for cr, cc in cells):
                continue
            candidates.append((r, c, rid, desired_rot_color))

    if not candidates:
        return None
    lr, lc, rid, marker_color = random.choice(candidates)
    for dr, dc in ROTATION_L_OFFSETS:
        input_grid[lr + dr, lc + dc] = marker_color
        output_grid[lr + dr, lc + dc] = out_colors[rid]
    return marker_color


def _rotation_is_visible(base_output: np.ndarray, rot_color: int | None) -> bool:
    if rot_color is None:
        return True
    rotated = np.rot90(base_output, ROTATION_BY_COLOR[rot_color])
    if np.array_equal(rotated, base_output):
        return False
    if rot_color == 8 and np.array_equal(rotated, np.rot90(base_output, 1)):
        return False
    return True


def _has_visible_region_one(input_grid: np.ndarray) -> bool:
    labels, n = label_mask(input_grid == 1, connectivity=4)
    for k in range(1, int(n) + 1):
        if np.count_nonzero(labels == k) >= MIN_REGION_CELLS:
            return True
    return False


def _gen_attempt(desired_rot_color: int | None, include_colored_center: bool) -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    directed_pairs = _find_adjacent_pairs(region)

    counts = _choose_marker_counts(n_regions, include_colored_center)
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

    input_colors = _choose_input_colors(n_regions)
    if input_colors is None:
        return None

    out_colors = list(input_colors)
    normal_pairs = marker_pairs[:n_normal]
    colored_pairs = marker_pairs[n_normal:]

    colored_targets = []
    for _src, dst in colored_pairs:
        cands = [c for c in REGION_COLOR_POOL if c != input_colors[dst]]
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
        center, stem = random.choice(cands)
        _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, tc))

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    rot_color = _try_place_rotation_marker(
        region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors, desired_rot_color
    )
    if rot_color != desired_rot_color:
        return None
    base_output = output_grid.copy()
    if not _rotation_is_visible(base_output, rot_color):
        return None
    if rot_color is not None:
        output_grid = np.rot90(base_output, ROTATION_BY_COLOR[rot_color])

    if not _has_visible_region_one(input_grid):
        return None

    try:
        result = main(input_grid)
    except Exception:
        return None
    return input_grid if np.array_equal(result, output_grid) else None


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT
    desired_rot_color = ROTATION_MODE_SCHEDULE[_GENERATE_CALL_COUNT % len(ROTATION_MODE_SCHEDULE)]
    include_colored_center = COLORED_CENTER_SCHEDULE[_GENERATE_CALL_COUNT % len(COLORED_CENTER_SCHEDULE)]
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(desired_rot_color, include_colored_center)
        if result is not None:
            _GENERATE_CALL_COUNT += 1
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
