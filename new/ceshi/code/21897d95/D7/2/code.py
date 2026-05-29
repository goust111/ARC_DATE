# Variant of 21897d95
# Dimension: D7 — 旋转触发机制（L9标记） (grid_shape)
# Change: 可选5单元L形颜色9标记：┐(右上角)触发逆时针90度旋转，┌(左上角)触发顺时针90度旋转；约50%概率出现... → 镜像翻转替代旋转
#
# Design analysis for this variant
# - Concrete behavior: the T-marker recoloring system is unchanged. The only semantic
#   change is the post-processing controlled by the optional 5-cell color-9 L marker.
#   The same two observed L orientations are kept: a top-right corner ("┐") still means
#   "apply a global transform", but now that transform is a left-right mirror; a top-left
#   corner ("┌") means an up-down mirror. The L marker is still erased into its host region
#   before the global transform is applied.
# - Small traced example: in the original task, a recolored 12x16 mosaic with a top-right
#   L would become 16x12 after a CCW rotation. In this variant, the same solved mosaic stays
#   12x16 and is mirrored horizontally instead. This changes both the output geometry and the
#   placement of all recolored regions and erased marker cells.
# - Complexity calibration: following the reflection source task 7ed72f31, the new branch is
#   still a single, visually legible mirror operation rather than a larger rule family. To keep
#   the difference observable, the generator rejects L9 cases where the mirrored output would be
#   identical to the untransformed output or accidentally identical to the legacy rotated output.
# - Hidden assumptions reviewed:
#   1. `_solve_unrotated()` already removes the L marker as a small irregular scrap, so the
#      transform must be applied only after that function returns.
#   2. `_find_l9_orientation()` assumes the L is a standalone 5-cell color-9 component inside a
#      3x3 box. The generator therefore forbids placing L9 on a region whose visible input color
#      is already 9, otherwise the marker would merge with the host region and become unreadable.
#   3. `main()` used to rely on `np.rot90`, which changed rectangular grid shape. Mirror flips do
#      not change shape, so no additional remapping logic is needed elsewhere.
# - Placement legality: T markers still reserve their cells and neighbor buffers. The L marker is
#   placed wholly inside one region, away from T cells and protected singleton branches, and only
#   when it is a distinct visible 9-shape that the solver can isolate as one 5-cell component.
# - Visual observability / cleanliness: flips are easier to verify by eye than rotations because
#   the frame size stays fixed. The generator keeps irregular region boundaries and multiple T
#   markers, which makes left-right and up-down mirror directions distinguishable in a few examples.
# - New solver assumptions checked against generation: if `main()` sees `top_right`, it now expects
#   a horizontal mirror target; if it sees `top_left`, it expects a vertical mirror target. The
#   generator explicitly builds outputs with those same mappings and oracle-checks them with `main`.

from __future__ import annotations

from collections import Counter, deque
import numpy as np

from arc_utils import label_mask
import random


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


def _apply_l9_transform(grid: np.ndarray, orient: str | None) -> np.ndarray:
    if orient == 'top_right':
        return np.fliplr(grid)
    if orient == 'top_left':
        return np.flipud(grid)
    return grid


def _apply_legacy_rotation(grid: np.ndarray, orient: str | None) -> np.ndarray:
    if orient == 'top_right':
        return np.rot90(grid, 1)
    if orient == 'top_left':
        return np.rot90(grid, 3)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor mosaic regions from T controls, erase the control shapes, and optionally mirror the finished picture from a color-9 L marker.

    Input structure:
    - The canvas is partitioned into monochrome regions with irregular borders.
    - Color 1 forms T-shaped boundary controls; some controls instead have a colored center with three singleton color-1 branches.
    - An optional 5-cell color-9 L shape sits inside one region and serves only as a global transform trigger.

    Rule:
    - For each T, the lone stem points to the destination region, while the two diagonal cells behind the stem identify the source side.
    - A plain color-1 T repaints the destination with the source region's color; a colored-center T repaints it with that center color.
    - Every control cell is then filled with the final color of its source-side region.
    - If a color-9 L marker is present, apply a whole-grid mirror after all recoloring: ┐ (corner at top-right) means horizontal flip, ┌ (corner at top-left) means vertical flip.

    Key insight:
    - The T geometry explains the recoloring, and the L geometry only chooses the final mirror direction; the two mechanisms stay independent until the last step.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    return _apply_l9_transform(out, _find_l9_orientation(g))

# --------------- generator (l9flip) ---------------

# ── Tunable constants ──────────────────────────────────────────────────
# Canvas / region sizing
GRID_SIZE_MAX = 30                          # max rows or cols
MIN_REGION_CELLS = 6                        # per-spec minimum
REGION_SIZE_FACTOR = 2.5                    # min_grid_dim = max(9, n_regions * this)
REGION_COUNTS = [4, 5, 6, 7, 8]            # candidate region counts
REGION_WEIGHTS = [3, 3, 3, 1, 1]           # sampling weights for REGION_COUNTS

# Marker counts (spec: 4-8 total, 2-6 normal, 0-2 colored)
MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

# Color pool (all ARC colors except 1, which is reserved for T-marker cells)
COLOR_POOL = [c for c in range(10) if c != 1]

# L9 mirror marker
L9_PROBABILITY = 0.5                        # chance of placing a mirror trigger
L9_OFFSETS = {                              # 5-cell L offsets keyed by corner orientation
    'top_right': [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    'top_left':  [(0, 0), (0,  1), (0,  2), (1, 0), (2, 0)],
}

# Generation retry budget
SEED_ATTEMPTS = 200                         # tries to place n distinct region seeds
GEN_RETRIES = 300                           # outer generate_input retry budget

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ── Small geometry helpers ─────────────────────────────────────────────

def _get_bar(stem):
    """Return the two bar directions perpendicular to *stem*."""
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center, stem):
    """Return the 4 cells of a T marker: center + stem tip + 2 bar cells."""
    r, c = center
    bar = _get_bar(stem)
    return {(r, c), (r + stem[0], c + stem[1]),
            (r + bar[0][0], c + bar[0][1]), (r + bar[1][0], c + bar[1][1])}


def _t_branch_cells(center, stem):
    """Return the 3 branch cells (stem tip + 2 bar), excluding the center."""
    r, c = center
    bar = _get_bar(stem)
    return {(r + stem[0], c + stem[1]),
            (r + bar[0][0], c + bar[0][1]),
            (r + bar[1][0], c + bar[1][1])}


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


# ── Region partitioning ────────────────────────────────────────────────

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

    # Validate: each region must be 4-connected and have enough cells
    for rid in range(n_regions):
        mask = region == rid
        if mask.sum() < MIN_REGION_CELLS:
            return None
        _, n = label_mask(mask, connectivity=4)
        if n != 1:
            return None
    return region


# ── Adjacency discovery ───────────────────────────────────────────────

def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    """Return all directed (src, dst) pairs between adjacent regions."""
    H, W = region.shape
    # Detect boundaries via horizontal and vertical neighbor differences
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


# ── Marker count selection ─────────────────────────────────────────────

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


# ── Marker placement search ───────────────────────────────────────────

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
                # Diagonal cells (stem + each bar dir) must lie in source region
                if not all(
                    0 <= r + stem[0] + bd[0] < H and
                    0 <= c + stem[1] + bd[1] < W and
                    region[r + stem[0] + bd[0], c + stem[1] + bd[1]] == src
                    for bd in bar
                ):
                    continue
                # Walk along stem from step=2 until we hit dst or leave bounds
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
    if kind == 'normal':
        occupied_cells |= cells
        occupied_neighbors |= (_t_neighborhood(center, stem) - cells)
    else:
        branch = _t_branch_cells(center, stem)
        occupied_cells |= branch
        occupied_cells.add(center)
        # Each branch cell must remain a singleton: buffer all non-marker neighbors
        for br, bc in branch:
            for dr, dc in _DIRS:
                nb = (br + dr, bc + dc)
                if nb != center and nb not in branch:
                    occupied_neighbors.add(nb)
        for dr, dc in _DIRS:
            nb = (center[0] + dr, center[1] + dc)
            if nb not in branch:
                occupied_neighbors.add(nb)


# ── Grid construction ─────────────────────────────────────────────────

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


# ── L9 rotation marker ────────────────────────────────────────────────

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
            if any(int(input_grid[cr, cc]) == 9 for cr, cc in cells):
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


def _transform_is_variant_visible(base_grid: np.ndarray, orient: str | None) -> bool:
    if orient is None:
        return True
    flipped = _apply_l9_transform(base_grid, orient)
    if np.array_equal(flipped, base_grid):
        return False
    legacy = _apply_legacy_rotation(base_grid, orient)
    if flipped.shape == legacy.shape and np.array_equal(flipped, legacy):
        return False
    return True


# ── Main generation attempt ───────────────────────────────────────────

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

    # Select src->dst pairs with unique destinations
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

    # Assign input colors
    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    # Compute output colors from recoloring rules
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
    for (src, dst), tc in zip(colored_pairs, colored_targets):
        out_colors[dst] = tc

    # Build marker definitions and place them
    all_defs = [('normal', s, d, None) for s, d in normal_pairs] + \
               [('colored', s, d, tc) for (s, d), tc in zip(colored_pairs, colored_targets)]
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

    # Build grids via vectorized lookup
    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    # Optional L9 mirror marker
    l9_orient = _try_place_l9(region, placed_markers, occupied_neighbors,
                              input_grid, output_grid, out_colors)
    if not _transform_is_variant_visible(output_grid, l9_orient):
        return None
    output_grid = _apply_l9_transform(output_grid, l9_orient)

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
