# Variant of 21897d95
# Dimension: D13 — T标记放置位置约束 (meta)
# Change: T标记放置在两个区域的边界上，茎跨越边界指向目标区域，对角线单元在源区域中... → T标记在目标区域内部

from __future__ import annotations

from collections import Counter
import numpy as np

from arc_utils import label_mask
import random


# Design analysis for the D13 variant
# -----------------------------------
# variant_spec:
#   core_change:
#     original:
#       - T marker sat on the source-side boundary.
#       - The stem direction was used to walk into the destination region.
#       - The source color was read from the diagonals on the stem side.
#     new:
#       - The T marker is stamped fully inside the destination region.
#       - The destination is now the region that surrounds the T marker locally.
#       - The stem points outward: the first non-marker cell beyond the stem is not in the
#         destination region.
#       - The source color is read from the two diagonals on the side opposite the stem.
#   concrete_pixel_behavior:
#     - For stem d and bar directions b1,b2, the four T cells are still center, center+d,
#       center+b1, center+b2.
#     - In this variant all four of those cells originally belong to the destination region.
#     - The source witness cells move from center+d+b to center-d+b.
#     - Generator also requires center-d to belong to the same source region. This is not
#       used by the solver as an explicit rule; it is a generation-side clarity constraint
#       that makes the opposite-side source visually coherent and keeps the two diagonal
#       witnesses in one obvious region.
#   behavior_trace:
#     - Original local pattern with stem RIGHT:
#         destination = first region met at (center + 2*RIGHT, center + 3*RIGHT, ...)
#         source      = cells at center + RIGHT + UP and center + RIGHT + DOWN
#     - New local pattern with stem RIGHT:
#         destination = region that touches most orthogonal neighbors of the T cells
#         source      = cells at center - RIGHT + UP and center - RIGHT + DOWN
#         extra visual constraint = cell center + 2*RIGHT is outside the destination
#     - So the solver must reverse the ownership inference:
#         old code followed the stem to decide "what changes";
#         new code first asks "which region contains this marker?", then uses the
#         opposite-side diagonals to decide "which color gets copied in".
#   complexity_calibration:
#     - The variant keeps the same small vocabulary: normal T, colored-center T, optional L9.
#     - Only one semantic axis changes, so examples stay readable.
#     - To avoid ambiguity, generation forces the destination to be the unique dominant
#       orthogonal neighbor around each T marker.
#   hidden_assumption_audit:
#     - The old line `dest = first_component_from(center, stem, start=2)` no longer holds,
#       because the stem does not identify the destination anymore.
#     - The old source lookup `center + stem + bar` is invalid under the inversion and must
#       become `center - stem + bar`.
#     - The old source-side union through `adjacent_component_counts(marker["cells"])`
#       depended on the marker touching its owner region orthogonally. That is now the
#       destination-side behavior, so this same-color merge must move to the destination.
#     - Marker filling still uses the source's final color; that assumption survives.
#   placement_legality:
#     - Generator enforces no overlap between markers and keeps branch singletons isolated
#       for colored markers, as in the original task.
#     - It additionally enforces:
#         1. all T cells lie in the destination region,
#         2. opposite-side diagonals lie in the source region,
#         3. the cell just beyond the stem leaves the destination,
#         4. the destination is the unique most-adjacent region around the T.
#   observability:
#     - In the original task, T markers sit on boundary transitions.
#     - In this variant, each T is visibly embedded in the region that will be repainted,
#       while its source clues are offset to the back side. Across 2-4 examples this creates
#       a stable, human-visible reversal: "the marker is inside what changes".
#   cleanliness:
#     - No extra object families are introduced.
#     - Region colors stay distinct before recoloring, so local ownership is visually clean.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
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


def _source_diagonal_cells(center: tuple[int, int], stem: tuple[int, int], bar: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (int(center[0] - stem[0] + bar_dir[0]), int(center[1] - stem[1] + bar_dir[1]))
        for bar_dir in bar
    ]


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
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        markers.append({"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells})
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
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "T marker must be embedded in a destination region"
            dest = max(counts, key=lambda c: (counts[c], comp_areas[c]))
            for comp in counts:
                if comp_colors[comp] == comp_colors[dest]:
                    dsu.union(dest, comp)

            source_parts: list[int] = []
            for rr, cc in _source_diagonal_cells(center, stem, marker["bar"]):
                assert _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0
                source_parts.append(int(comp_id[rr, cc]))
            source = source_parts[0]
            for part in source_parts[1:]:
                assert comp_colors[part] == comp_colors[source], "T source diagonals must share a color"
                dsu.union(source, part)

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
                    return 'top_right'
                if cr == r0 and cc == c0:
                    return 'top_left'
                assert False, f"unobserved L9 corner at ({cr},{cc})"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor regions from embedded T controls, erase those controls, and optionally rotate when a color-9 L appears.

    Input structure:
    - The grid is partitioned into monochrome regions plus control markers.
    - Color 1 still forms T-shaped controls; some controls instead use a colored center with three singleton color-1 branches.
    - Unlike the base task, each T sits inside the region that will be repainted.
    - An optional 5-cell L-shaped color-9 marker still requests a 90-degree rotation of the final image.

    Rule:
    - For either kind of T, the region surrounding the marker is the destination region to recolor.
    - The stem only gives orientation: the two diagonals on the side opposite the stem identify the source region.
    - A normal T copies the source region's original color into the destination; a colored-center T uses the center color instead.
    - Every control cell is overwritten by the final color of the source-side region.
    - A color-9 L marker keeps its old role: ┐ rotates the output 90 degrees counterclockwise, and ┌ rotates it 90 degrees clockwise.

    Key insight:
    - The marker no longer points to what changes; instead, the marker lives inside what changes, and the stem must be read backwards to find the color donor.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == 'top_right':
        return np.rot90(out, 1)
    if l9 == 'top_left':
        return np.rot90(out, 3)
    return out


# --------------- generator (l9rot) ---------------

# ── Tunable constants ──────────────────────────────────────────────────
# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 4.0
REGION_COUNTS = [4, 5, 6]
REGION_WEIGHTS = [3, 3, 2]

# Marker counts (spec: 4-8 total, 2-6 normal, 0-2 colored)
MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

# Color pool (all ARC colors except 1, which is reserved for T-marker cells)
COLOR_POOL = [c for c in range(10) if c != 1]

# L9 rotation marker
L9_PROBABILITY = 0.5
L9_OFFSETS = {
    'top_right': [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    'top_left': [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

# Generation retry budget
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


def _region_adjacent_counts(region: np.ndarray, cells: set[tuple[int, int]]) -> Counter:
    counts: Counter = Counter()
    H, W = region.shape
    for r, c in cells:
        for dr, dc in _DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cells:
                counts[int(region[rr, cc])] += 1
    return counts


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    """Build connected, slightly jagged stripe regions that reliably support embedded T markers."""
    min_band = 4

    def build_stripes(rows: int, cols: int) -> np.ndarray | None:
        if cols < n_regions * min_band:
            return None
        widths = [min_band] * n_regions
        for _ in range(cols - n_regions * min_band):
            widths[random.randrange(n_regions)] += 1
        boundaries = [0]
        for w in widths:
            boundaries.append(boundaries[-1] + w)
        all_bounds = [boundaries]
        for _ in range(1, rows):
            prev = all_bounds[-1]
            curr = prev[:]
            for i in range(1, n_regions):
                curr[i] += random.choice([-1, 0, 1])
            for i in range(1, n_regions):
                curr[i] = max(curr[i], curr[i - 1] + min_band)
            for i in range(n_regions - 1, 0, -1):
                curr[i] = min(curr[i], curr[i + 1] - min_band)
            curr[0] = 0
            curr[-1] = cols
            all_bounds.append(curr)

        region = np.full((rows, cols), -1, dtype=int)
        for r, bounds in enumerate(all_bounds):
            for rid in range(n_regions):
                region[r, bounds[rid]:bounds[rid + 1]] = rid
        return region

    horizontal = random.random() < 0.5
    if horizontal:
        region = build_stripes(W, H)
        if region is None:
            return None
        region = region.T
    else:
        region = build_stripes(H, W)
        if region is None:
            return None

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
    """Find all valid (center, stem) for placing a T marker inside dst while reading src from the back side."""
    region_areas = np.bincount(region.ravel())
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in _DIRS:
                bar = _get_bar(stem)
                cells = _t_cells((r, c), stem)
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                if not all(int(region[cr, cc]) == dst for cr, cc in cells):
                    continue
                if cells & occupied_cells:
                    continue
                if _t_neighborhood((r, c), stem) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue

                source_diags = [(r - stem[0] + bd[0], c - stem[1] + bd[1]) for bd in bar]
                if not all(
                    0 <= rr < H and 0 <= cc < W and int(region[rr, cc]) == src
                    for rr, cc in source_diags
                ):
                    continue

                back_mid = (r - stem[0], c - stem[1])
                if not (0 <= back_mid[0] < H and 0 <= back_mid[1] < W and int(region[back_mid]) == src):
                    continue

                step = 2
                leaves_dst = False
                while True:
                    nr, nc = r + step * stem[0], c + step * stem[1]
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if int(region[nr, nc]) != dst:
                        leaves_dst = True
                        break
                    step += 1
                if not leaves_dst:
                    continue

                counts = _region_adjacent_counts(region, cells)
                if not counts or dst not in counts:
                    continue
                owner = max(counts, key=lambda rid: (counts[rid], int(region_areas[rid])))
                if owner != dst:
                    continue

                candidates.append(((r, c), stem))
    return candidates


def _register_marker(kind, center, stem, occupied_cells, occupied_neighbors):
    """Update occupied_cells and occupied_neighbors after placing a marker."""
    cells = _t_cells(center, stem)
    if kind == 'normal':
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
    for kind, center, stem, _src, _dst, tc in placed_markers:
        if kind == 'normal':
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = tc
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(grid, placed_markers, out_colors):
    """Fill marker cells in the output grid with the source region's final color."""
    for kind, center, stem, src, _dst, _tc in placed_markers:
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
    for kind, center, stem, *_ in placed_markers:
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

    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    n_markers, n_normal, n_colored = counts

    feasible_counts: dict[tuple[int, int], int] = {}
    for src, dst in directed_pairs:
        cands = _find_valid_placements(region, src, dst, set(), set(), H, W)
        if cands:
            feasible_counts[(src, dst)] = len(cands)
    if not feasible_counts:
        return None

    pair_order = list(feasible_counts)
    random.shuffle(pair_order)
    pair_order.sort(key=lambda pair: feasible_counts[pair], reverse=True)

    base_pairs: list[tuple[int, int]] = []
    dest_used = set()
    capacity = 0
    for pair in pair_order:
        src, dst = pair
        if dst in dest_used:
            continue
        base_pairs.append(pair)
        dest_used.add(dst)
        capacity += feasible_counts[pair]
        if len(base_pairs) >= 2 and capacity >= n_markers:
            break
    if capacity < MARKER_MIN or not base_pairs:
        return None

    n_markers = min(n_markers, capacity)
    if n_markers < MARKER_MIN:
        return None
    n_colored = min(n_colored, COLORED_MAX, len(base_pairs), n_markers - 2)
    n_normal = n_markers - n_colored
    if n_normal < 2:
        return None

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    out_colors = list(input_colors)
    colored_pairs = random.sample(base_pairs, n_colored) if n_colored else []
    normal_base_pairs = [pair for pair in base_pairs if pair not in colored_pairs]
    if not normal_base_pairs:
        return None

    colored_targets = []
    for _src, dst in colored_pairs:
        cands = [c for c in pool if c != input_colors[dst]]
        if not cands:
            return None
        colored_targets.append(random.choice(cands))

    for src, dst in normal_base_pairs:
        out_colors[dst] = input_colors[src]
    for (_src, dst), tc in zip(colored_pairs, colored_targets):
        out_colors[dst] = tc

    all_defs = [('normal', s, d, None) for s, d in normal_base_pairs] + [
        ('colored', s, d, tc) for (s, d), tc in zip(colored_pairs, colored_targets)
    ]
    normal_usage: Counter = Counter(normal_base_pairs)
    remaining_normals = n_normal - len(normal_base_pairs)
    while remaining_normals > 0:
        available = [pair for pair in normal_base_pairs if normal_usage[pair] < feasible_counts[pair]]
        if not available:
            return None
        random.shuffle(available)
        available.sort(key=lambda pair: (feasible_counts[pair] - normal_usage[pair], feasible_counts[pair]), reverse=True)
        pair = available[0]
        all_defs.append(('normal', pair[0], pair[1], None))
        normal_usage[pair] += 1
        remaining_normals -= 1

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers = []
    pending_defs = list(all_defs)
    while pending_defs:
        best_idx = -1
        best_cands = None
        for idx, (kind, src, dst, tc) in enumerate(pending_defs):
            cands = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W)
            if not cands:
                return None
            if best_cands is None or len(cands) < len(best_cands):
                best_idx = idx
                best_cands = cands
        kind, src, dst, tc = pending_defs.pop(best_idx)
        center, stem = random.choice(best_cands)
        _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, tc))

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    l9_orient = _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors)
    if l9_orient == 'top_right':
        output_grid = np.rot90(output_grid, 1)
    elif l9_orient == 'top_left':
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
