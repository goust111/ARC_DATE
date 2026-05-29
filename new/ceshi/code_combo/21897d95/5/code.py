# Combo variant of 21897d95 (index 5)
# combo_spec: D7/2+D8/0
# Name: 四向镜像系统
#
# Design analysis for this combo
# - Concretization / behavior trace:
#   - The base task still has one main stage: solve all T-shaped recoloring controls on the
#     irregular region mosaic, erase the control markers into their source-side colors, and
#     only then inspect the optional color-9 L marker.
#   - The fused change is entirely about that final L9 stage. The marker is still a 5-cell
#     elbow inside a 3x3 box, but now all four elbow positions are legal and they no longer
#     mean quarter-turn rotations. Instead:
#       `top_right`  (┐) -> horizontal flip,
#       `top_left`   (┌) -> vertical flip,
#       `bottom_right` (┘) -> horizontal + vertical flip,
#       `bottom_left`  (└) -> identity; the marker is present but intentionally does nothing.
#   - Small trace: suppose `_solve_untransformed()` builds a 12x17 recolored mosaic `O`.
#     Original task with a top-right L would output `rot90(O)`, changing shape to 17x12.
#     This combo keeps shape 12x17 and outputs `fliplr(O)` instead. A bottom-right L now
#     yields `flipud(fliplr(O))`, while a bottom-left L keeps `O` unchanged despite the
#     marker still appearing in the input.
# - Complexity calibration:
#   - The T logic is untouched so the problem does not balloon into two interacting local
#     rule systems. The added complexity is a controlled 4-way symbol vocabulary on one
#     object family.
#   - Only four discrete L states exist, and three of them have visibly distinct whole-grid
#     effects. The fourth is an explicit distractor state whose meaning is still geometric,
#     not color-based.
#   - The generator cycles through all four L orientations instead of randomly omitting the
#     marker, and it rejects transformed cases whose output accidentally equals the
#     unflipped base or another non-identity mirror. That keeps the orientation mapping
#     learnable from just a few examples instead of collapsing under symmetry.
# - Hidden assumptions audit:
#   - `_solve_untransformed()` already absorbs every small non-rectangular component,
#     including a valid L9, into its owner region color. Therefore the L-trigger must still
#     be applied after that function returns.
#   - The original `_find_l9_orientation()` assumed only top-corner elbows existed; here it
#     must classify all four normalized 3x3 elbow templates.
#   - The original generator could place only top-corner L shapes and sometimes inside a
#     visible color-9 region. That latter case would merge the marker with its host region
#     and make the instruction unreadable, so this combo explicitly forbids such placements.
#   - `main()` now assumes every instructive L9 is an isolated 5-cell, 4-connected, 3x3-
#     bounded component. `generate_input()` enforces exactly that.
# - Placement legality / cleanliness:
#   - T markers still reserve both their own cells and the singleton buffers needed by
#     colored-center T branches.
#   - L candidates must lie wholly inside one region, avoid all T cells, avoid the branch
#     protection buffer, and avoid pre-existing visible color-9 cells.
#   - Because the L is erased before the flip stage, the output mosaic stays clean: no
#     overdrawn symbols survive, only recolored regions.
# - Visual observability / inferability:
#   - Horizontal, vertical, and double flips keep the same canvas size, making the global
#     transform easy to compare against the untransformed recolored mosaic.
#   - The `bottom_left` case is intentionally a no-op distractor, but the input still shows
#     a valid L, so solvers must infer that elbow orientation matters rather than merely the
#     presence of color 9.
#   - No rule depends on numeric color ordering. Colors only identify regions and literal
#     replacement centers, which are both directly visible.
# - Generator guarantees for new solver assumptions:
#   - If `main()` branches on `top_right`, `top_left`, or `bottom_right`, the generated
#     target is guaranteed to differ from the identity result and from the other mirror
#     branches on that same base output.
#   - If `main()` sees `bottom_left`, the generator deliberately allows the identity branch;
#     that is the intended semantics, not degeneration.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

L9_NORMALIZED_SHAPES = {
    "top_left": frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)}),
    "top_right": frozenset({(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)}),
    "bottom_left": frozenset({(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)}),
    "bottom_right": frozenset({(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}),
}
L9_OFFSETS = {
    orient: sorted(shape) for orient, shape in L9_NORMALIZED_SHAPES.items()
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


def _solve_untransformed(g: np.ndarray) -> np.ndarray:
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

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"
    dsu = _DSU(len(comp_colors))

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
            for part in source_parts[1:]:
                assert (
                    comp_colors[part] == comp_colors[source]
                ), "T source diagonals must share a color"
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
        replacement = (
            int(marker["target_color"])
            if marker["kind"] == "colored"
            else int(root_color[source])
        )
        assert final_color[dest] in {
            root_color[dest],
            replacement,
        }, "conflicting T controls for one region"
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
    """Find an isolated 5-cell color-9 L marker and return its elbow orientation."""
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    for k in range(1, int(n) + 1):
        coords = np.argwhere(labels == k)
        if len(coords) != 5:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1) - int(r0) != 2 or int(c1) - int(c0) != 2:
            continue
        normalized = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
        for orient, shape in L9_NORMALIZED_SHAPES.items():
            if normalized == shape:
                return orient
    return None


def _apply_l9_transform(grid: np.ndarray, orient: str | None) -> np.ndarray:
    if orient == "top_right":
        return np.fliplr(grid)
    if orient == "top_left":
        return np.flipud(grid)
    if orient == "bottom_right":
        return np.flipud(np.fliplr(grid))
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve the T-based recoloring first, then let an optional color-9 L marker choose a whole-grid mirror transform.

    Input structure:
    - The board is partitioned into solid-color regions with irregular boundaries.
    - Color 1 makes T controls on region borders, either as a full 4-cell T or as three color-1 branches around a colored center.
    - An optional 5-cell color-9 L lies fully inside one region, fits in a 3x3 box, and its elbow may be at any of the four corners.

    Rule:
    - For each T control, the unique stem points to the destination region, while the two diagonal cells behind that stem identify the source side.
    - A plain T repaints the destination with the source region's original color; a colored-center T uses the center color instead.
    - After recoloring, every control cell is filled with the final color of its source-side region.
    - If an L9 marker exists, apply a final global mirror chosen by its elbow: ┐ means horizontal flip, ┌ means vertical flip, ┘ means both flips, and └ means leave the solved grid as-is.

    Key insight:
    - The T geometry determines all local color changes, while the L geometry is a separate late-stage instruction whose elbow orientation alone selects one of four whole-grid mirror outcomes, including an explicit no-op case.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_untransformed(g)
    return _apply_l9_transform(out, _find_l9_orientation(g))


# --------------- generator (l9mirror4) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

COLOR_POOL = [c for c in range(10) if c != 1]

L9_ORIENTATION_SEQUENCE = ("top_right", "top_left", "bottom_right", "bottom_left")
_generate_call_count = 0

SEED_ATTEMPTS = 200
GEN_RETRIES = 300

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


def _try_place_l9(
    region,
    placed_markers,
    occupied_neighbors,
    input_grid,
    output_grid,
    out_colors,
    required_orient: str | None,
):
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, stem, *_ in placed_markers:
        all_marker_cells |= _t_cells(center, stem)

    orient_candidates: list[tuple[str, list[tuple[int, int, int]]]] = []
    for orient, offsets in L9_OFFSETS.items():
        if required_orient is not None and orient != required_orient:
            continue
        candidates = []
        for r in range(H - 2):
            for c in range(W - 2):
                cells = [(r + dr, c + dc) for dr, dc in offsets]
                rid = int(region[cells[0]])
                if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                    continue
                if any(int(input_grid[cr, cc]) == 9 for cr, cc in cells):
                    continue
                if any((cr, cc) in all_marker_cells for cr, cc in cells):
                    continue
                if any((cr, cc) in occupied_neighbors for cr, cc in cells):
                    continue
                candidates.append((r, c, rid))
        if candidates:
            orient_candidates.append((orient, candidates))

    if not orient_candidates:
        return None
    orient, candidates = random.choice(orient_candidates)
    lr, lc, rid = random.choice(candidates)
    for dr, dc in L9_OFFSETS[orient]:
        input_grid[lr + dr, lc + dc] = 9
        output_grid[lr + dr, lc + dc] = out_colors[rid]
    return orient


def _transform_is_variant_visible(base_grid: np.ndarray, orient: str | None) -> bool:
    if orient in {None, "bottom_left"}:
        return True
    transformed = _apply_l9_transform(base_grid, orient)
    if np.array_equal(transformed, base_grid):
        return False
    for other in ("top_right", "top_left", "bottom_right", None, "bottom_left"):
        if other == orient:
            continue
        if np.array_equal(transformed, _apply_l9_transform(base_grid, other)):
            return False
    return True


def _gen_attempt(required_orient: str | None) -> np.ndarray | None:
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

    l9_orient = _try_place_l9(
        region,
        placed_markers,
        occupied_neighbors,
        input_grid,
        output_grid,
        out_colors,
        required_orient,
    )
    if l9_orient != required_orient:
        return None
    if not _transform_is_variant_visible(output_grid, l9_orient):
        return None
    output_grid = _apply_l9_transform(output_grid, l9_orient)

    try:
        result = main(input_grid)
    except Exception:
        return None
    return input_grid if np.array_equal(result, output_grid) else None


def generate_input() -> np.ndarray:
    global _generate_call_count
    required_orient = L9_ORIENTATION_SEQUENCE[
        _generate_call_count % len(L9_ORIENTATION_SEQUENCE)
    ]
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(required_orient)
        if result is not None:
            _generate_call_count += 1
            return result
    raise RuntimeError(
        f"Failed to generate valid input after {GEN_RETRIES} attempts for {required_orient}"
    )
