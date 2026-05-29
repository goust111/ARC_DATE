# Variant of 21897d95
# Dimension: D5 — normal T 与 colored T 的区分机制 (symbol_instructions)
# Change: normal T：4个颜色1单元；colored T：中心为非1颜色+3个单独颜色1邻居，中心颜色为替换色... → T标记朝向区分行为

from __future__ import annotations

from collections import Counter
import numpy as np

from arc_utils import label_mask
import random


# Design analysis for this D5 variant
# a) Concrete behavior:
#    - Every control is now the same visual object: a 4-cell color-1 T tetromino.
#    - The stem orientation alone chooses the instruction family.
#    - Stem up/down => normal behavior: repaint the pointed destination region to the
#      source region color found from the two diagonals behind the stem.
#    - Stem left/right => colored behavior: repaint the pointed destination region to
#      the color of the unique third region touching the marker. The generator makes
#      that third region sit on the open back side of the T, so the extra color is
#      visible as a real spatial cue instead of a hidden id-based rule.
#    - Example trace:
#      original colored-T: center pixel might be color 6, three singleton 1-neighbors
#      form a T, so destination becomes 6.
#      new horizontal T: all four T cells are 1; source is still read from front
#      diagonals, destination is still read from the stem direction, but the output
#      color comes from the extra region touching the T (the third touched color),
#      not from a colored center cell.
# b) Complexity calibration:
#    - If horizontal markers were allowed to touch arbitrary many regions, the "third
#      color" would become ambiguous. The generator therefore forces horizontal T
#      markers to touch exactly three regions: source, destination, and one extra
#      region. Vertical markers are constrained to touch only source and destination,
#      making the contrast visible across a few examples.
#    - At least one horizontal marker is generated in every sample so the new branch
#      is actually observable.
# c) Hidden assumptions changed in main():
#    - The original scan for colored-center markers is invalid because center color is
#      no longer a discriminator; all color-1 cells must now belong to 4-cell T
#      tetrominoes.
#    - The original replacement = marker["target_color"] for colored markers is also
#      invalid. Replacement must instead be derived from adjacency after component
#      labeling, and the solver must assert that exactly one extra touched region
#      exists for a horizontal T.
#    - The source/destination parsing code remains valid because the stem and front
#      diagonals still carry the same semantics.
# d) Placement legality:
#    - Markers may not overlap or touch each other.
#    - Vertical markers only use up/down stems; horizontal markers only use left/right
#      stems.
#    - Horizontal markers are only placed where the touched region set is exactly
#      {source, destination, third}, with the back-side cell belonging to that third
#      region. This guarantees that the solver's uniqueness assertion is satisfied.
# e) Visual observability:
#    - Humans can compare vertical and horizontal T markers directly because the shape
#      is identical and only orientation changes.
#    - Horizontal markers visibly sit at a three-region contact, so the extra color
#      they invoke is readable from the grid rather than from an invisible metadata
#      choice.
# f) Visual cleanliness:
#    - Marker neighborhoods are buffered so T markers stay separated.
#    - Vertical markers avoid third-region contact, reducing clutter and preventing
#      "why did this extra color not matter here?" confusion.
# g) Color-change inferability:
#    - Region colors remain few and distinct.
#    - A horizontal marker always changes its destination to a color different from the
#      source region's input color, because the third region id must differ from both
#      source and destination ids and region input colors are unique.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
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


def _parse_tetromino(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
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

    markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)

    for cells in one_components.values():
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        kind = "colored" if stem[0] == 0 else "normal"
        markers.append({"kind": kind, "center": center, "stem": stem, "bar": bar, "cells": cells})
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

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

            adj_counts = adjacent_component_counts(marker["cells"])
            for comp in adj_counts:
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
            marker["dest_comp"] = dest
            marker["source_comp"] = source
            marker["adjacent_comps"] = list(adj_counts)
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
        if marker["kind"] == "normal":
            replacement = int(root_color[source])
        else:
            extra_roots = {
                dsu.find(comp)
                for comp in marker["adjacent_comps"]
                if dsu.find(comp) not in {source, dest}
            }
            assert len(extra_roots) == 1, "horizontal T must touch exactly one third region"
            replacement = int(root_color[next(iter(extra_roots))])
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
    Repaint regions from orientation-coded T markers, erase the markers, and optionally rotate based on a color-9 L marker.

    Input structure:
    - The grid is partitioned into monochrome regions.
    - Every color-1 control is a 4-cell T tetromino placed on region boundaries; there is no longer a special colored center form.
    - Vertical-stem T markers and horizontal-stem T markers share the same shape, so their absolute orientation is the only instruction cue.
    - An optional 5-cell L-shaped color-9 marker still indicates a 90-degree output rotation.

    Rule:
    - The stem points toward the destination region, and the two front diagonals behind the stem still identify the source side of the marker.
    - If the stem points up or down, the destination is repainted to the source region's original color.
    - If the stem points left or right, the destination is repainted to the color of the unique third region touching that T marker.
    - All T-marker cells are filled with the final color of the source-side region, and the L marker rotates the finished grid: ┐ means CCW 90 degrees, ┌ means CW 90 degrees.

    Key insight:
    - Shape no longer separates instruction families; only orientation does, so you must read both where the T points and whether it is vertical or horizontal.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (l9rot + orientation-coded T semantics) ---------------

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
COLORED_MIN = 1
COLORED_MAX = 2

# Color pool (all ARC colors except 1, which is reserved for T-marker cells)
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


def _t_neighborhood(center, stem):
    cells = _t_cells(center, stem)
    neighborhood = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _adjacent_region_ids(region: np.ndarray, cells: set[tuple[int, int]]) -> set[int]:
    H, W = region.shape
    touched: set[int] = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cells:
                touched.add(int(region[nr, nc]))
    return touched


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
    max_colored = min(COLORED_MAX, n_markers - 2)
    if max_colored < COLORED_MIN:
        return None
    n_colored = random.randint(COLORED_MIN, max_colored)
    n_normal = n_markers - n_colored
    if n_normal > NORMAL_MAX:
        n_colored = n_markers - NORMAL_MAX
        n_normal = NORMAL_MAX
    if not (2 <= n_normal <= NORMAL_MAX and COLORED_MIN <= n_colored <= COLORED_MAX):
        return None
    return n_markers, n_normal, n_colored


def _find_valid_placements(region, kind, src, dst, occupied_cells, occupied_neighbors, H, W):
    candidates = []
    stems = (UP, DOWN) if kind == "normal" else (LEFT, RIGHT)
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in stems:
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
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == src
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
                if not found:
                    continue
                touched = _adjacent_region_ids(region, cells)
                if kind == "normal":
                    if touched != {src, dst}:
                        continue
                    candidates.append(((r, c), stem, None))
                    continue
                back_r, back_c = r - stem[0], c - stem[1]
                if not (0 <= back_r < H and 0 <= back_c < W):
                    continue
                third = int(region[back_r, back_c])
                if third in {src, dst}:
                    continue
                if touched != {src, dst, third}:
                    continue
                candidates.append(((r, c), stem, third))
    return candidates


def _register_marker(center, stem, occupied_cells, occupied_neighbors):
    cells = _t_cells(center, stem)
    occupied_cells |= cells
    occupied_neighbors |= (_t_neighborhood(center, stem) - cells)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid, placed_markers):
    for _kind, center, stem, _src, _dst, _rep in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = 1


def _stamp_markers_output(grid, placed_markers, out_colors):
    for _kind, center, stem, src, _dst, _rep in placed_markers:
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
    for _kind, center, stem, *_ in placed_markers:
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


def _place_markers(region: np.ndarray, n_normal: int, n_colored: int):
    H, W = region.shape
    directed_pairs = _find_adjacent_pairs(region)
    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    dest_used = set()
    placed_markers = []

    for kind in (["colored"] * n_colored) + (["normal"] * n_normal):
        pairs = directed_pairs[:]
        random.shuffle(pairs)
        feasible = []
        for src, dst in pairs:
            if dst in dest_used:
                continue
            cands = _find_valid_placements(region, kind, src, dst, occupied_cells, occupied_neighbors, H, W)
            if cands:
                feasible.append((src, dst, cands))
        if not feasible:
            return None
        feasible.sort(key=lambda item: len(item[2]))
        src, dst, cands = random.choice(feasible[: min(4, len(feasible))])
        center, stem, rep = random.choice(cands)
        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, rep))
        dest_used.add(dst)

    return placed_markers, occupied_neighbors


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    _n_markers, n_normal, n_colored = counts

    placed = _place_markers(region, n_normal, n_colored)
    if placed is None:
        return None
    placed_markers, occupied_neighbors = placed

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    out_colors = list(input_colors)
    for kind, _center, _stem, src, dst, rep in placed_markers:
        out_colors[dst] = input_colors[src] if kind == "normal" else input_colors[int(rep)]

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
