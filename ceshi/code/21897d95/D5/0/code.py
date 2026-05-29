# Variant of 21897d95
# Dimension: D5 — normal T 与 colored T 的区分机制 (symbol_instructions)
# Change: normal T：4个颜色1单元；colored T：中心为非1颜色+3个单独颜色1邻居，中心颜色为替换色... → 所有T均为normal，替换色由图例区域定义

# Design analysis for this D5 variant
# -----------------------------------
# Reference note:
# - The prompt cites `dbff022c` as the source-task baseline for palette legends.
# - Within the allowed files, `dbff022c.py` is not present; only
#   `ARC-data/ARC-AGI-2/data/evaluation/dbff022c.json` is available.
# - The implementation therefore uses the visible 2xN edge-palette idea from those
#   official I/O examples, while keeping all non-D5 mechanics from 21897d95 intact.
#
# a) Concrete behavior and flow impact
# - Original control split:
#   - normal T: 4 connected color-1 cells, destination becomes the source region color.
#   - colored T: three singleton color-1 neighbors around a colored center, and that
#     center color overrides the destination.
# - Variant control split:
#   - every T is now the same 4-cell color-1 tetromino;
#   - the override color is no longer stored in the T itself;
#   - instead, a top-edge 2xN legend lists `(source_color, replacement_color)` pairs
#     column by column.
# - Concrete pixel form:
#   - the legend is a 2-row strip on the top edge, not touching corners;
#   - it is carved out of a single owner region, so in the output the whole legend is
#     erased back to that region's final color;
#   - T geometry, source-diagonal lookup, small-scrap absorption, and optional L9
#     rotation remain unchanged.
# - Small traced example:
#   - original: a colored T with source color 6 and colored center 3 repaints its
#     destination to 3 directly.
#   - variant: the same boundary location contains a normal all-1 T; the top legend
#     contains a column `6 / 3`; the solver first decodes `6 -> 3`, then when the T
#     identifies source color 6 it uses 3 as the destination color.
#   - difference: the recolor instruction moved from local marker appearance to a
#     separate symbol-reading stage on the grid edge.
#
# b) Complexity calibration
# - The legend uses 2-4 columns. Fewer than 2 columns makes the extra decoding step
#   too weak; many more columns crowd the edge and reduce placement success.
# - Each sample uses at least one non-identity recolor, so the legend has a visible
#   effect and the task cannot collapse back to pure source-color copying.
# - The legend is always a compact rectangle on the top edge. This keeps the decoding
#   family small and learnable from 2-4 examples instead of introducing many edge-
#   dependent reading conventions at once.
#
# c) Implicit-assumption audit against the original main()
# - The original non-1 scan looked for colored-center T markers. That assumption no
#   longer holds and would misclassify legend cells; the scan is removed.
# - The original `replacement = target_color if colored else source_color` assumed the
#   override color lived inside each marker. In the variant, replacement must come from
#   `legend[source_color]`, where `source_color` is the source region's original input
#   color (`root_color[source]`), not its possibly recolored final color.
# - The original control mask marked only T markers and small scraps. Without adding
#   the legend to `control`, legend cells would become bogus base regions and break the
#   recoloring logic. The solver now detects the legend before base-component labeling.
# - The original small-scrap pass was harmless because colored-T centers were already
#   consumed. In the variant the legend must also be consumed first, otherwise its tiny
#   colored cells would leak into component labeling.
#
# d) Placement legality
# - Generator constraints guarantee the legend occupies rows 0-1 of a single region,
#   with row 2 and the left/right neighbors still showing the owner region color.
# - T markers cannot overlap the legend and are buffered away from it by one cell so
#   parsing remains visually clean.
# - The optional L9 marker is placed with the same no-overlap constraint and continues
#   to be erased as a small embedded control shape.
#
# e) Visual observability
# - Because all T markers now look identical, the legend must explain every recolor.
# - The top row always lists visible source colors, and the second row gives their
#   replacement colors. This is directly inferable from a few samples by matching the
#   source-side region of each T with the destination color in the output.
# - The generator forces at least one true color change per sample, so the legend is
#   necessary to solve the task.
#
# f) Visual cleanliness
# - The legend lives in a single compact strip on the edge instead of being scattered.
# - T markers are kept away from the legend and from each other with the same spacing
#   discipline as the original generator, so region boundaries stay legible.
#
# g) Color-change inferability
# - Region colors remain distinct and come from a small pool with color 1 reserved for
#   T cells and color 9 reserved for the optional L marker. This avoids hidden color-id
#   tricks and keeps the legend readable.
# - The legend columns are explicit pairings, so the human-visible structure tells the
#   solver which source color triggers which replacement color.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

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


def _find_top_legend(g: np.ndarray) -> dict:
    H, W = g.shape
    assert H >= 3 and W >= 4, "legend variant requires at least 3 rows and 4 columns"
    candidates: list[dict] = []
    max_width = min(4, W - 2)
    for width in range(2, max_width + 1):
        for c0 in range(1, W - width):
            c1 = c0 + width - 1
            owner = int(g[2, c0])
            if any(int(g[2, c]) != owner for c in range(c0, c1 + 1)):
                continue
            if int(g[0, c0 - 1]) != owner or int(g[1, c0 - 1]) != owner:
                continue
            if int(g[0, c1 + 1]) != owner or int(g[1, c1 + 1]) != owner:
                continue
            pairs = []
            ok = True
            for c in range(c0, c1 + 1):
                top = int(g[0, c])
                bottom = int(g[1, c])
                if top in {1, 9} or bottom in {1, 9}:
                    ok = False
                    break
                if top == owner or bottom == owner:
                    ok = False
                    break
                pairs.append((top, bottom))
            if not ok:
                continue
            cells = {(r, c) for r in range(2) for c in range(c0, c1 + 1)}
            candidates.append(
                {
                    "kind": "legend",
                    "cells": cells,
                    "owner_color_hint": owner,
                    "pairs": pairs,
                    "mapping": {src: dst for src, dst in pairs},
                }
            )
    assert len(candidates) == 1, "expected exactly one top-edge legend"
    legend = candidates[0]
    assert len(legend["mapping"]) == len(legend["pairs"]), "legend source colors must be unique"
    return legend


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
        markers.append({"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells})
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a normal T marker"

    legend = _find_top_legend(g)
    markers.append(legend)
    for r, c in legend["cells"]:
        control[r, c] = True

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
        if marker["kind"] == "normal":
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
            continue

        counts = adjacent_component_counts(marker["cells"])
        assert counts, f"{marker['kind']} marker must touch its owner region"
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
    legend_map = legend["mapping"]
    for marker in markers:
        if marker["kind"] != "normal":
            marker["source"] = dsu.find(marker["source_comp"])
            continue
        dest = dsu.find(marker["dest_comp"])
        source = dsu.find(marker["source_comp"])
        marker["source"] = source
        source_color = int(root_color[source])
        assert source_color in legend_map, "legend must define every T source color"
        replacement = int(legend_map[source_color])
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
    Repaint regions from uniform T markers, using a separate edge legend to decode the replacement colors, and optionally rotate from a color-9 L marker.

    Input structure:
    - The grid is an irregular mosaic of monochrome regions.
    - Every T control is a 4-cell color-1 tetromino placed on a region boundary; there are no colored-center T markers in this variant.
    - A compact 2xN legend sits on the top edge. Each column is a `(source color, replacement color)` pair read from top cell to bottom cell.
    - An optional 5-cell L-shaped color-9 marker may also be embedded inside one region.

    Rule:
    - For each T marker, the single stem points toward the destination region.
    - The two diagonal cells behind the stem identify the source region exactly as in the original task.
    - The destination does not copy the source color directly. Instead, look up the source region's input color in the top-edge legend and repaint the destination to the legend's replacement color.
    - All control symbols are erased in the output: T cells are filled with the source region's final color, the legend is filled back with its owner region's final color, and the L marker is likewise erased after determining any rotation.
    - A color-9 L marker rotates the finished grid: ┐ means 90° counterclockwise, ┌ means 90° clockwise.

    Key insight:
    - The T still tells you which region acts on which other region, but the actual paint color is no longer local to the marker. You must first decode the shared legend, then apply that color mapping to every T source color.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (legend_t) ---------------

# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

# Marker counts: all T markers are now the same visual type
MARKER_MIN = 4
MARKER_MAX = 8

# Legend sizing
LEGEND_MIN = 2
LEGEND_MAX = 4

# Color pool: 1 is reserved for T cells, 9 is reserved for the optional L marker
COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

# L9 rotation marker
L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

# Generation retry budget
SEED_ATTEMPTS = 200
GEN_RETRIES = 400
PAIR_RETRIES = 200

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


def _legend_cells(start_col: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(2) for c in range(start_col, start_col + width)}


def _legend_neighborhood(start_col: int, width: int, H: int, W: int) -> set[tuple[int, int]]:
    cells = _legend_cells(start_col, width)
    out = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cells:
                out.add((rr, cc))
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


def _find_legend_slots(region: np.ndarray) -> list[tuple[int, int, int]]:
    H, W = region.shape
    slots = []
    if H < 3 or W < 4:
        return slots
    for width in range(LEGEND_MIN, min(LEGEND_MAX, W - 2) + 1):
        for c0 in range(1, W - width):
            c1 = c0 + width - 1
            owner = int(region[0, c0])
            if any(int(region[r, c]) != owner for r in range(3) for c in range(c0, c1 + 1)):
                continue
            if any(int(region[r, c0 - 1]) != owner for r in range(3)):
                continue
            if any(int(region[r, c1 + 1]) != owner for r in range(3)):
                continue
            slots.append((owner, c0, width))
    return slots


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
                if found:
                    candidates.append(((r, c), stem))
    return candidates


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _choose_marker_pairs(directed_pairs, source_rids, n_regions):
    src_set = set(source_rids)
    candidates = [(src, dst) for src, dst in directed_pairs if src in src_set]
    feasible_dests = {dst for _src, dst in candidates}
    min_markers = max(MARKER_MIN, len(source_rids))
    max_markers = min(MARKER_MAX, len(feasible_dests), n_regions)
    if max_markers < min_markers:
        return None
    for _ in range(PAIR_RETRIES):
        n_markers = random.randint(min_markers, max_markers)
        chosen = []
        used_dests = set()
        ok = True

        shuffled_sources = list(source_rids)
        random.shuffle(shuffled_sources)
        for src in shuffled_sources:
            opts = [(s, d) for s, d in candidates if s == src and d not in used_dests]
            if not opts:
                ok = False
                break
            pair = random.choice(opts)
            chosen.append(pair)
            used_dests.add(pair[1])
        if not ok:
            continue

        remaining = [(s, d) for s, d in candidates if d not in used_dests]
        random.shuffle(remaining)
        for pair in remaining:
            if len(chosen) >= n_markers:
                break
            chosen.append(pair)
            used_dests.add(pair[1])

        if len(chosen) == n_markers:
            random.shuffle(chosen)
            return chosen
    return None


def _choose_replacements(source_colors: list[int], owner_color: int) -> dict[int, int] | None:
    available = [c for c in COLOR_POOL if c != owner_color]
    for _ in range(PAIR_RETRIES):
        repls = []
        random.shuffle(available)
        for src in source_colors:
            choices = [c for c in available if c != src and c not in repls]
            if not choices:
                break
            repls.append(random.choice(choices))
        if len(repls) == len(source_colors):
            return {src: dst for src, dst in zip(source_colors, repls)}
    return None


def _register_marker(center, stem, occupied_cells, occupied_neighbors):
    cells = _t_cells(center, stem)
    occupied_cells |= cells
    occupied_neighbors |= (_t_neighborhood(center, stem) - cells)


def _stamp_t_markers_input(grid, placed_markers):
    for center, stem, _src, _dst in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = 1


def _stamp_t_markers_output(grid, placed_markers, out_colors):
    for center, stem, src, _dst in placed_markers:
        color = out_colors[src]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _stamp_legend_input(grid, start_col: int, pairs: list[tuple[int, int]]):
    for idx, (src_color, dst_color) in enumerate(pairs):
        c = start_col + idx
        grid[0, c] = src_color
        grid[1, c] = dst_color


def _stamp_legend_output(grid, start_col: int, width: int, owner_final_color: int):
    for r in range(2):
        for c in range(start_col, start_col + width):
            grid[r, c] = owner_final_color


def _try_place_l9(region, occupied_cells, occupied_neighbors, input_grid, output_grid, out_colors):
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    candidates = []
    for r in range(H):
        for c in range(W):
            cells = [(r + dr, c + dc) for dr, dc in offsets]
            if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if any((cr, cc) in occupied_cells for cr, cc in cells):
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
        occupied_cells.add((lr + dr, lc + dc))
    return orient


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    legend_slots = _find_legend_slots(region)
    if not legend_slots:
        return None

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    random.shuffle(legend_slots)
    directed_pairs = _find_adjacent_pairs(region)
    plan = None
    for owner_rid, start_col, width in legend_slots:
        source_candidates = [rid for rid in range(n_regions) if rid != owner_rid]
        if len(source_candidates) < width:
            continue
        for _ in range(PAIR_RETRIES):
            source_rids = random.sample(source_candidates, width)
            marker_pairs = _choose_marker_pairs(directed_pairs, source_rids, n_regions)
            if marker_pairs is None:
                continue
            source_colors = [input_colors[rid] for rid in source_rids]
            replacement_map = _choose_replacements(source_colors, input_colors[owner_rid])
            if replacement_map is None:
                continue
            plan = (owner_rid, start_col, width, source_rids, marker_pairs, replacement_map)
            break
        if plan is not None:
            break
    if plan is None:
        return None

    owner_rid, start_col, width, source_rids, marker_pairs, replacement_map = plan
    legend_pairs = [(input_colors[rid], replacement_map[input_colors[rid]]) for rid in source_rids]

    out_colors = list(input_colors)
    for src, dst in marker_pairs:
        out_colors[dst] = replacement_map[input_colors[src]]

    occupied_cells: set[tuple[int, int]] = set(_legend_cells(start_col, width))
    occupied_neighbors: set[tuple[int, int]] = _legend_neighborhood(start_col, width, H, W)
    placed_markers = []
    for src, dst in marker_pairs:
        cands = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W)
        if not cands:
            return None
        center, stem = random.choice(cands)
        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((center, stem, src, dst))

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_legend_input(input_grid, start_col, legend_pairs)
    _stamp_legend_output(output_grid, start_col, width, out_colors[owner_rid])
    _stamp_t_markers_input(input_grid, placed_markers)
    _stamp_t_markers_output(output_grid, placed_markers, out_colors)

    l9_orient = _try_place_l9(region, occupied_cells, occupied_neighbors, input_grid, output_grid, out_colors)
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
