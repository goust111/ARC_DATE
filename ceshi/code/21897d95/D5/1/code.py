# Variant of 21897d95
# Dimension: D5 — normal T 与 colored T 的区分机制 (symbol_instructions)
# Change: normal T：4个颜色1单元；colored T：中心为非1颜色+3个单独颜色1邻居，中心颜色为替换色... → T标记大小区分行为

from __future__ import annotations

from collections import Counter
import numpy as np

from arc_utils import label_mask
import random


# Design analysis for the D5 variant:
# a) Concrete behavior:
#    - Small T stays the original 4-cell all-1 tetromino: it copies the source region color into the destination.
#    - Colored behavior is no longer inferred from "non-1 center + three singleton 1s". Instead the marker is a
#      visibly larger 6-cell T centered on a colored hub. The hub still carries the literal replacement color, but
#      the solver must first notice that the T is larger before using that payload.
#    - Two large geometries are allowed: a wide T (longer bar) or a tall T (longer stem). The local stem/bar junction
#      is unchanged, so the source diagonals are still read from center + stem + bar; only the marker extent changes.
#      Example: if a tall large T is centered at (5,5) with stem downward, its 1-cells are at (6,5),(7,5),(8,5),
#      (5,4),(5,6). The source still comes from (6,4) and (6,6), but the destination search now begins at (9,5)
#      instead of (7,5) because the stem itself is longer.
# b) Complexity calibration:
#    - The only new control variable is marker size: 4 cells means "copy source color"; 6 cells means "use hub color".
#    - To keep this visible, generation forces at least one large marker per grid. Large markers use a single extension
#      style per grid so the puzzle does not mix too many visual dialects at once.
# c) Hidden assumptions in the original solver that must change:
#    - Original colored-T parsing depended on the center having exactly three singleton color-1 neighbors.
#      That is false here because large markers contain 5 color-1 cells and some form 2- or 3-cell chains.
#    - Original destination search always started two steps along the stem. That is only correct for the 4-cell T and
#      for the wide large T; the tall large T needs a longer start offset.
#    - Original "all 1s are markers" assertion remains valid, but used_ones must now collect the five 1-cells of a
#      large T, not just singleton branches around a colored center.
# d) Placement legality:
#    - Generator uses the same boundary logic for source/destination as before.
#    - It buffers the whole marker neighborhood, not just singleton branches, so large T arms cannot touch other
#      markers or accidentally create extra color-1 adjacencies that would confuse parsing.
#    - Solver and generator share the same geometry helpers and destination-start constants.
# e) Visual observability:
#    - A human can still read the stem and the two source diagonals exactly as before.
#    - The new cue is geometric and visible: some T markers are plainly larger, and those are the only ones whose hub
#      color becomes the destination color.
# f) Visual cleanliness:
#    - Large markers are only 6 cells, so they remain compact. The one-cell buffer keeps markers separated from each
#      other and from the optional L9 rotation cue.
# g) Color inference:
#    - Region colors remain few and distinct. The hub color only matters on large T markers, and the size distinction
#      gives a direct structural hint for when that literal color should override the copied region color.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BIG_T_VARIANTS = ("wide", "long")
BIG_T_DEST_START = {"wide": 2, "long": 4}


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


def _add(cell: tuple[int, int], direction: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return (int(cell[0] + direction[0] * scale), int(cell[1] + direction[1] * scale))


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if int(stem[0]) == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _stem_from_branches(branches: list[tuple[int, int]]) -> tuple[int, int]:
    assert len(branches) == 3
    branch_set = set(branches)
    for d in branches:
        if _opp(d) not in branch_set:
            return d
    raise AssertionError("T marker has no single stem")


def _parse_small_t(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
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


def _big_t_one_cells(
    center: tuple[int, int],
    stem: tuple[int, int],
    variant: str,
) -> set[tuple[int, int]]:
    bar = _get_bar(stem)
    cells = {_add(center, stem), _add(center, bar[0]), _add(center, bar[1])}
    if variant == "wide":
        cells.add(_add(center, bar[0], 2))
        cells.add(_add(center, bar[1], 2))
        return cells
    if variant == "long":
        cells.add(_add(center, stem, 2))
        cells.add(_add(center, stem, 3))
        return cells
    raise ValueError(f"unknown large-T variant {variant}")


def _marker_cells(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    variant: str | None = None,
) -> set[tuple[int, int]]:
    if kind == "normal":
        bar = _get_bar(stem)
        return {center, _add(center, stem), _add(center, bar[0]), _add(center, bar[1])}
    assert variant is not None
    return {center} | _big_t_one_cells(center, stem, variant)


def _marker_one_cells(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    variant: str | None = None,
) -> set[tuple[int, int]]:
    cells = _marker_cells(kind, center, stem, variant)
    cells.remove(center)
    return cells


def _marker_neighborhood(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    variant: str | None = None,
) -> set[tuple[int, int]]:
    cells = _marker_cells(kind, center, stem, variant)
    neighborhood = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


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


def _parse_large_t_at(
    g: np.ndarray,
    center: tuple[int, int],
    used_ones: set[tuple[int, int]],
) -> dict | None:
    r, c = center
    branches = []
    for d in DIRS:
        rr, cc = _add(center, d)
        if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1 and (rr, cc) not in used_ones:
            branches.append(d)
    if len(branches) != 3:
        return None

    stem = _stem_from_branches(branches)
    bar = [d for d in branches if d != stem]
    assert len(bar) == 2 and bar[0] == _opp(bar[1])

    for variant in BIG_T_VARIANTS:
        one_cells = _big_t_one_cells(center, stem, variant)
        if not all(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1 and (rr, cc) not in used_ones for rr, cc in one_cells):
            continue
        if variant == "wide":
            forbidden = [_add(center, stem, 2), _add(center, stem, 3)]
        else:
            forbidden = [_add(center, bar[0], 2), _add(center, bar[1], 2)]
        if any(_in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1 for rr, cc in forbidden):
            continue
        return {
            "stem": stem,
            "bar": bar,
            "variant": variant,
            "one_cells": one_cells,
            "cells": one_cells | {center},
            "dest_start": int(BIG_T_DEST_START[variant]),
        }
    return None


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
        parsed = _parse_small_t(cells)
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
                "dest_start": 2,
            }
        )
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            parsed = _parse_large_t_at(g, (int(r), int(c)), used_ones)
            if parsed is None:
                continue
            markers.append(
                {
                    "kind": "colored",
                    "center": (int(r), int(c)),
                    "stem": parsed["stem"],
                    "bar": parsed["bar"],
                    "variant": parsed["variant"],
                    "cells": parsed["cells"],
                    "dest_start": parsed["dest_start"],
                    "target_color": int(g[r, c]),
                }
            )
            used_ones |= parsed["one_cells"]
            for rr, cc in parsed["cells"]:
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
            dest = first_component_from(center, stem, start=marker["dest_start"])
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
    Repaint regions from size-coded T markers, erase the markers, and optionally rotate from a color-9 L cue.

    Input structure:
    - The board is partitioned into monochrome regions.
    - Color 1 draws T-shaped boundary markers.
    - A 4-cell T is the small form; a 6-cell T is the large form, and the large form has a colored hub cell.
    - An optional 5-cell L marker in color 9 requests a 90-degree rotation at the end.

    Rule:
    - For every T, the unique stem direction points toward the destination region.
    - The two diagonal cells behind the first stem segment identify the source-side region.
    - Small 4-cell T: repaint the destination with the source region's original color.
    - Large 6-cell T: repaint the destination with the hub cell's literal color instead.
    - Every marker cell is finally filled with the source-side region's final color.
    - A color-9 L marker rotates the finished board: top-right corner means CCW 90 degrees, top-left means CW 90 degrees.

    Key insight:
    - The stem and diagonals are read the same way for every T; the only question is whether the T is small or large.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (l9rot-sizecoded) ---------------

# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

# Marker counts. At least one colored marker is required so the D5 change is observable.
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
    max_colored = min(COLORED_MAX, n_markers - 2)
    if max_colored < COLORED_MIN:
        return None
    n_colored = random.randint(COLORED_MIN, max_colored)
    n_normal = n_markers - n_colored
    if n_normal > NORMAL_MAX:
        n_colored = n_markers - NORMAL_MAX
        n_normal = NORMAL_MAX
        if n_colored > COLORED_MAX or n_colored < COLORED_MIN:
            n_markers = NORMAL_MAX + COLORED_MAX
            if n_markers > max_possible:
                return None
            n_colored = COLORED_MAX
            n_normal = NORMAL_MAX
    return n_markers, n_normal, n_colored


def _marker_destination_start(kind: str, variant: str | None) -> int:
    if kind == "normal":
        return 2
    assert variant is not None
    return int(BIG_T_DEST_START[variant])


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    kind: str,
    variant: str | None,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Find all valid (center, stem) placements for one marker."""
    candidates = []
    for r in range(H):
        for c in range(W):
            center = (r, c)
            for stem in _DIRS:
                bar = _get_bar(stem)
                cells = _marker_cells(kind, center, stem, variant)
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                if cells & occupied_cells:
                    continue
                if _marker_neighborhood(kind, center, stem, variant) & occupied_cells:
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

                step = _marker_destination_start(kind, variant)
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
                    candidates.append((center, stem))
    return candidates


def _register_marker(marker: dict, occupied_cells: set[tuple[int, int]], occupied_neighbors: set[tuple[int, int]]) -> None:
    """Update occupied_cells and occupied_neighbors after placing a marker."""
    cells = _marker_cells(marker["kind"], marker["center"], marker["stem"], marker.get("variant"))
    occupied_cells |= cells
    occupied_neighbors |= (_marker_neighborhood(marker["kind"], marker["center"], marker["stem"], marker.get("variant")) - cells)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    """Map region ids to colors via vectorized lookup."""
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[dict]) -> None:
    """Write T-marker cells onto the input grid."""
    for marker in placed_markers:
        kind = marker["kind"]
        center = marker["center"]
        stem = marker["stem"]
        variant = marker.get("variant")
        if kind == "normal":
            for cr, cc in _marker_cells(kind, center, stem, variant):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = int(marker["target_color"])
            for cr, cc in _marker_one_cells(kind, center, stem, variant):
                grid[cr, cc] = 1


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[dict], out_colors: list[int]) -> None:
    """Fill marker cells in the output grid with the source region's final color."""
    for marker in placed_markers:
        color = int(out_colors[marker["src"]])
        for cr, cc in _marker_cells(marker["kind"], marker["center"], marker["stem"], marker.get("variant")):
            grid[cr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[dict],
    occupied_neighbors: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
) -> str | None:
    """Attempt to place a 5-cell L9 marker; return orientation string or None."""
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for marker in placed_markers:
        all_marker_cells |= _marker_cells(marker["kind"], marker["center"], marker["stem"], marker.get("variant"))

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
        max_colored = min(COLORED_MAX, n_markers - 2)
        if max_colored < COLORED_MIN:
            return None
        n_colored = min(n_colored, max_colored)
        n_colored = max(n_colored, COLORED_MIN)
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

    colored_style = random.choice(BIG_T_VARIANTS)
    all_defs = (
        [{"kind": "normal", "src": s, "dst": d, "variant": None, "target_color": None} for s, d in normal_pairs]
        + [
            {"kind": "colored", "src": s, "dst": d, "variant": colored_style, "target_color": tc}
            for (s, d), tc in zip(colored_pairs, colored_targets)
        ]
    )
    random.shuffle(all_defs)

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers: list[dict] = []
    for marker in all_defs:
        cands = _find_valid_placements(
            region,
            marker["src"],
            marker["dst"],
            marker["kind"],
            marker["variant"],
            occupied_cells,
            occupied_neighbors,
            H,
            W,
        )
        if not cands:
            return None
        center, stem = random.choice(cands)
        placed = dict(marker)
        placed["center"] = center
        placed["stem"] = stem
        _register_marker(placed, occupied_cells, occupied_neighbors)
        placed_markers.append(placed)

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
