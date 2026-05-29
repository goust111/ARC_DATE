# Combo variant of 21897d95 (index 7)
# combo_spec: D2/3+D9/0
# Name: 洞与碎片共存

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# ---------------------------------------------------------------------------
# Design analysis for combo D2/3 + D9/0
#
# concretization_and_trace:
#   - The base task already has three semantic layers in the input:
#     large monochrome regions, T-shaped recolor controls, and an optional
#     color-9 L that requests a 90-degree rotation.
#   - This combo adds a fourth input-time category that stays visible at
#     output time: a tiny non-rectangular scrap embedded inside one region.
#   - Concrete trace on a small local situation:
#       1. Suppose a blue T points into a yellow destination while the two
#          diagonal source cells lie in a green region. Base logic recolors the
#          yellow region to green.
#       2. In the original task, the four T cells were then painted green too,
#          so the control visually merged into the source side.
#       3. Under D2/3, those same four T cells become color 0 instead, leaving
#          a T-shaped hole on the boundary after recoloring.
#       4. Suppose the same grid also contains a tiny orange 3-cell L inside a
#          large cyan region. Base logic would absorb that chip into cyan.
#       5. Under D9/0, that orange chip is excluded from recoloring and written
#          back as orange, so the final output simultaneously shows recolored
#          large regions, preserved tiny scraps, and 0-colored holes.
#   - The key difference from either single variant is coexistence: the solver
#     must classify tiny non-rectangular components into "special L9", "true
#     scrap", and "not a component at all", while also treating T writeback
#     differently from both.
#
# complexity_calibration:
#   - To make the combo visually obvious, the generator reserves color 0 only
#     for holes and strongly enforces one preserved scrap whenever possible.
#   - Only one scrap is used, so the new burden is categorical reasoning rather
#     than bookkeeping across many distractors.
#   - T behavior remains uniform across normal and colored-center controls:
#     every T becomes a hole. This keeps the transform crisp and repeated.
#
# implicit_assumption_review:
#   - Base `_solve_unrotated` assumed every small non-rectangular component
#     should join an adjacent region. That is false now; only the actual color-9
#     L marker should be absorbed.
#   - Base final writeback assumed every marker-like object inherits a region
#     color. That is now split three ways:
#       * T markers -> HOLE_COLOR
#       * L9 marker -> owner region final color
#       * scraps -> their own preserved input color
#   - Base component labeling also assumed no special object survives outside
#     the recolored regions. The combo requires scraps to be masked out before
#     region analysis, then restored afterward.
#
# placement_legality:
#   - T markers still cannot overlap, touch each other too closely, or leave the
#     grid bounds. Colored-center T branches are buffered so each color-1 arm is
#     a singleton.
#   - Scraps are placed fully inside one host region, with a one-cell buffer of
#     host-region cells around them, so they touch only their owner region and
#     remain easy to classify.
#   - Removing scrap cells from the host region must leave that host 4-connected
#     and above the minimum size. L9 is also kept disjoint from scraps and Ts.
#
# visual_observability_and_cleanliness:
#   - Human-visible output should contain three repeated, distinct motifs:
#       * large recolored regions,
#       * a small preserved chip,
#       * black T-shaped holes.
#   - Reserving 0 for holes prevents confusion between holes and ordinary region
#     colors. Keeping the scrap away from markers avoids clutter.
#   - The only tiny component that disappears is the color-9 L, which stays
#     distinguishable from scraps by both color and exact 5-cell shape.
#
# inferability_and_new_assumptions:
#   - The solver newly assumes that any preserved scrap is a <=5-cell
#     non-rectangular component that is not the L9 shape. The generator enforces
#     exactly that.
#   - The solver also assumes holes come only from Ts, so the generator forbids
#     color 0 for regions, scraps, and colored-center targets.
#   - Because the output must visibly demonstrate coexistence, generation fails
#     if it cannot place a preserved scrap.
# ---------------------------------------------------------------------------


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOLE_COLOR = 0


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


def _small_nonrect_component(coords: np.ndarray) -> bool:
    if len(coords) > 5:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return len(coords) != (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)


def _find_l9_marker(g: np.ndarray) -> tuple[str | None, set[tuple[int, int]]]:
    mask9 = g == 9
    if not mask9.any():
        return None, set()
    labels, n = label_mask(mask9, connectivity=4)
    found: tuple[str, set[tuple[int, int]]] | None = None
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
        for cr, cc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            h_nb = (cr, cc + 1 if cc == c0 else cc - 1)
            v_nb = (cr + 1 if cr == r0 else cr - 1, cc)
            if h_nb in coords and v_nb in coords:
                if cr == r0 and cc == c1:
                    orient = "top_right"
                elif cr == r0 and cc == c0:
                    orient = "top_left"
                else:
                    continue
                assert found is None, "multiple L9 markers are not supported"
                found = (orient, coords)
    if found is None:
        return None, set()
    return found


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        k: {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == k)}
        for k in range(1, int(n_1) + 1)
    }
    one_sizes = {k: len(cells) for k, cells in one_components.items()}

    markers: list[dict] = []
    scraps: list[dict] = []
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

    _l9_orient, l9_cells = _find_l9_marker(g)
    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == col) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == k)
            if not _small_nonrect_component(coords):
                continue
            cells = {tuple(map(int, xy)) for xy in coords}
            if cells == l9_cells:
                markers.append({"kind": "l9", "cells": cells})
            else:
                scraps.append({"cells": cells, "color": int(col)})
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
        elif marker["kind"] == "l9":
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "L9 marker must touch its owner region"
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
        if marker["kind"] == "l9":
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
        color = HOLE_COLOR if marker["kind"] in {"normal", "colored"} else int(
            final_color[marker["source"]]
        )
        for r, c in marker["cells"]:
            out[r, c] = color

    for scrap in scraps:
        for r, c in scrap["cells"]:
            out[r, c] = int(scrap["color"])

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def _find_l9_orientation(g: np.ndarray) -> str | None:
    orient, _cells = _find_l9_marker(g)
    return orient


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the large regions from T controls, turn those T cells into holes, preserve any true tiny scrap, and optionally rotate via a color-9 L marker.

    Input structure:
    - The canvas is mostly a partition of large monochrome regions with irregular borders.
    - Color 1 appears only as T-shaped control arms, either as a full 4-cell T or as three singleton 1-cells around a colored center.
    - There may be one extra tiny non-rectangular colored scrap inside a region; unlike the large regions, this chip is not part of the recoloring transfer.
    - An optional 5-cell color-9 L sits inside one region and still serves only as a rotation cue.

    Rule:
    - For each T, the unique stem direction points toward the destination region, and the two diagonally offset cells behind that stem identify the source side.
    - A normal T recolors the destination to the source region's color, while a colored-center T recolors it to the center color.
    - After recoloring, every T cell becomes background color 0, producing a 4-cell hole; the color-9 L blends into its owner region as before.
    - Any other tiny non-rectangular scrap stays exactly its input color in the output.
    - If the color-9 L has a top-right corner, rotate the finished output 90 degrees counterclockwise; if it has a top-left corner, rotate 90 degrees clockwise.

    Key insight:
    - Tiny objects must be split into two classes: the special 9-colored L controls rotation and disappears into its host region, while every other small scrap survives unchanged; meanwhile the T geometry still determines recoloring, but its own cells erase to holes.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (holes + preserved_scrap + l9rot) ---------------

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

# Reserve 0 for holes and 1 for T-marker arms.
COLOR_POOL = [c for c in range(10) if c not in {0, 1}]

# L9 rotation marker
L9_BRANCH_CHOICES = [None, "top_left", "top_right"]
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

# Preserved scrap
SCRAP_PROBABILITY = 1.0
SCRAP_SHAPES = [
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
]

# Generation retry budget
SEED_ATTEMPTS = 200
GEN_RETRIES = 500

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


def _neighbor_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                out.add(nb)
    return out


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


def _stamp_markers_output(grid, placed_markers):
    for _kind, center, stem, _src, _dst, _tc in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = HOLE_COLOR


def _try_place_l9(
    region,
    placed_markers,
    occupied_neighbors,
    blocked_cells,
    blocked_neighbors,
    input_grid,
    output_grid,
    out_colors,
):
    orient = random.choice(L9_BRANCH_CHOICES)
    if orient is None:
        return None, set()
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set(blocked_cells)
    for _kind, center, stem, *_rest in placed_markers:
        all_marker_cells |= _t_cells(center, stem)

    candidates = []
    for r in range(H):
        for c in range(W):
            cells = {(r + dr, c + dc) for dr, dc in offsets}
            if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if cells & all_marker_cells:
                continue
            if cells & occupied_neighbors:
                continue
            if cells & blocked_neighbors:
                continue
            candidates.append((rid, cells))

    if not candidates:
        return None, set()
    rid, cells = random.choice(candidates)
    for cr, cc in cells:
        input_grid[cr, cc] = 9
        output_grid[cr, cc] = out_colors[rid]
    return orient, cells


def _try_place_scrap(
    region: np.ndarray,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    input_colors: list[int],
    out_colors: list[int],
    blocked_cells: set[tuple[int, int]],
    blocked_neighbors: set[tuple[int, int]],
) -> dict | None:
    if random.random() >= SCRAP_PROBABILITY:
        return None

    H, W = region.shape
    region_sizes = np.bincount(region.ravel(), minlength=int(region.max()) + 1)
    candidates = []
    for shape in SCRAP_SHAPES:
        max_r = max(dr for dr, _dc in shape)
        max_c = max(dc for _dr, dc in shape)
        for r0 in range(H - max_r):
            for c0 in range(W - max_c):
                cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
                rid = int(region[r0, c0])
                if any(int(region[r, c]) != rid for r, c in cells):
                    continue
                if cells & blocked_cells:
                    continue
                neigh = _neighbor_cells(cells)
                if cells & blocked_neighbors:
                    continue
                if neigh & blocked_cells:
                    continue
                if neigh & blocked_neighbors:
                    continue
                if any(
                    not _in_bounds(region.shape, nr, nc)
                    or int(region[nr, nc]) != rid
                    for nr, nc in neigh
                ):
                    continue
                if int(region_sizes[rid]) - len(cells) < MIN_REGION_CELLS:
                    continue
                host_mask = region == rid
                for rr, cc in cells:
                    host_mask[rr, cc] = False
                _, n = label_mask(host_mask, connectivity=4)
                if n != 1:
                    continue
                color_choices = [
                    c
                    for c in COLOR_POOL
                    if c not in {9, input_colors[rid], out_colors[rid]}
                ]
                if not color_choices:
                    continue
                candidates.append(
                    {
                        "rid": rid,
                        "cells": cells,
                        "neighbors": neigh,
                        "color_choices": color_choices,
                    }
                )

    if not candidates:
        return None
    choice = random.choice(candidates)
    color = random.choice(choice["color_choices"])
    for r, c in choice["cells"]:
        input_grid[r, c] = color
        output_grid[r, c] = color
    return {"cells": choice["cells"], "color": color, "rid": choice["rid"]}


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
        ("colored", s, d, tc)
        for (s, d), tc in zip(colored_pairs, colored_targets)
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
    _stamp_markers_output(output_grid, placed_markers)

    l9_orient, l9_cells = _try_place_l9(
        region,
        placed_markers,
        occupied_neighbors,
        blocked_cells=set(),
        blocked_neighbors=set(),
        input_grid=input_grid,
        output_grid=output_grid,
        out_colors=out_colors,
    )

    extra_blocked_cells = set(occupied_cells) | set(l9_cells)
    extra_blocked_neighbors = set(occupied_neighbors) | _neighbor_cells(set(l9_cells))
    scrap = _try_place_scrap(
        region,
        input_grid,
        output_grid,
        input_colors,
        out_colors,
        extra_blocked_cells,
        extra_blocked_neighbors,
    )
    if scrap is None:
        return None

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
