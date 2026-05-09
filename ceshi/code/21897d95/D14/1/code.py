# Variant of 21897d95
# Dimension: D14 — 区域连通性要求 (meta)
# Change: 所有区域必须是4-连通的单连通区域... → 区域可以有洞（非单连通）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D14 variant
# -----------------------------------
# a) Concrete behavior:
# - Keep the original region-recoloring task, T markers, colored-center T markers,
#   and optional L9 rotation marker.
# - The visible novelty is that one region is an annulus/ring that surrounds a
#   second region. A T owned by the ring can point inward from the outer side of
#   the ring, so the stem ray first passes through more ring cells and only then
#   reaches the enclosed destination.
# - Small trace example:
#   original logic: first non-marker component on the stem ray is always the
#   destination.
#   variant logic: first non-marker component may still be the source region
#   itself (the ring). The solver must skip source-owned cells on the stem ray
#   and use the first different region after that as the destination.
#   difference: destination identification now depends on nested topology, not
#   just immediate adjacency.
# b) Complexity calibration:
# - Only one nested pair is forced per grid. This makes the change visible but
#   does not flood the puzzle with many independent special cases.
# - The ring is thick (4-5 cells per side), so the "through the outer region"
#   effect is visually obvious and not confused with the original boundary-touch
#   behavior.
# - Colors remain distinct per input region, so nested interpretation is carried
#   by geometry rather than hidden color-id conventions.
# c) Hidden assumptions in the original main():
# - `_label_base_components`: still safe. A region with a hole is still one
#   4-connected component after controls are removed.
# - `first_component_from(..., start=2)`: unsafe under D14. It assumes the first
#   component on the stem ray is always the destination. This fails when the ray
#   passes through a ring source before reaching the enclosed region.
# - `adjacent_component_counts(marker["cells"])`: still useful. It merges any
#   source fragments caused by marker carving, which is needed before tracing
#   through the ring.
# - Final painting logic remains valid once destination tracing is corrected.
# d) Placement legality:
# - The generator builds a ring + inner rectangle first, then partitions only
#   the outside free space. This guarantees one actual hole and avoids accidental
#   overlaps or disconnected outside regions.
# - The forced nested T requires at least two source cells on the stem ray before
#   the enclosed destination, so the example cannot silently collapse to the
#   original immediate-hit behavior.
# - Marker occupancy / neighbor buffering from the original generator is kept so
#   singleton color-1 branches and T silhouettes remain parseable.
# e) Visual observability:
# - Every generated grid contains one ring surrounding one inner region, and at
#   least one T whose stem tunnels through that ring to recolor the inner region.
# - This makes the D14 change visible in 2-3 examples without needing any
#   invisible ordering or color-id rules.
# f) Visual cleanliness:
# - The hole structure is one clean annulus. Other regions live outside it, so
#   nested geometry is easy to read.
# - The generator excludes color 9 from normal region palettes so the optional
#   L marker stays visually distinct.
# g) Color-change inferability:
# - Recoloring remains the original "source color or colored center color"
#   mapping; no new color arithmetic is introduced.
# - The only new inference burden is stem-ray traversal through a surrounding
#   region, which is directly visible in the pixel layout.
# New solver-side assumptions introduced here:
# - For the forced nested pair, generator guarantees the stem ray encounters the
#   source region again before the destination.
# - Input region colors are distinct before recoloring, so skipping "same source
#   root" on the ray is unambiguous.
# - The ring never touches the canvas border, leaving one connected outside area
#   for the remaining regions.


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

    def adjacent_component_counts(cells: set[tuple[int, int]]) -> Counter:
        counts: Counter = Counter()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0:
                    counts[int(comp_id[rr, cc])] += 1
        return counts

    def trace_first_non_source_component(
        center: tuple[int, int],
        direction: tuple[int, int],
        source_root: int,
        start: int = 2,
    ) -> int | None:
        r, c = center
        step = int(start)
        while True:
            rr = r + direction[0] * step
            cc = c + direction[1] * step
            if not _in_bounds(g.shape, rr, cc):
                return None
            cid = int(comp_id[rr, cc])
            if cid < 0:
                step += 1
                continue
            if dsu.find(cid) == int(source_root):
                step += 1
                continue
            return cid

    for marker in markers:
        if marker["kind"] in {"normal", "colored"}:
            center = marker["center"]
            stem = marker["stem"]

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

            source_root = dsu.find(source)
            dest = trace_first_non_source_component(center, stem, source_root, start=2)
            assert dest is not None, "T marker must point at a destination region"

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
                assert False, f"unobserved L9 corner at ({cr},{cc})"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint regions from T-shaped control markers, including nested regions hidden inside holes, and optionally rotate via a color-9 L marker.

    Input structure:
    - The grid is tiled by monochrome regions; one of them may be a ring that encloses another region.
    - Color 1 forms T-shaped control markers on region boundaries.
    - Some controls use three singleton color-1 cells around a colored center; that center gives a literal replacement color.
    - An optional 5-cell L-shaped color-9 marker sits inside a region and later controls a 90-degree output rotation.

    Rule:
    - The two diagonal cells behind the stem identify the source region that owns the marker.
    - Follow the stem ray forward: skip marker cells and also skip any cells that still belong to that same source region; the first different region on the ray is the destination to repaint.
    - A normal T repaints the destination to the source region's original color; a colored-center T repaints it to the center color.
    - All marker cells are filled with the final color of the source-side region; the L marker is also erased into its owner region before any optional rotation.

    Key insight:
    - Under this variant, the stem does not always stop at the first region it touches. If the source is a ring, the stem can tunnel through that ring and select the enclosed region inside the hole.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (D14 holes) ---------------

GRID_SIZE_MIN = 16
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_COUNTS = [4, 5, 6]
REGION_WEIGHTS = [3, 3, 2]

MARKER_MIN = 4
MARKER_MAX = 6
NORMAL_MAX = 6
COLORED_MAX = 2

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

GEN_RETRIES = 400
LAYOUT_RETRIES = 80
SEED_ATTEMPTS = 200
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _grow_regions_on_mask(mask: np.ndarray, n_regions: int) -> np.ndarray | None:
    region = np.full(mask.shape, -1, dtype=int)
    free_cells = [tuple(map(int, xy)) for xy in np.argwhere(mask)]
    if len(free_cells) < n_regions * MIN_REGION_CELLS:
        return None

    seeds: set[tuple[int, int]] = set()
    for _ in range(SEED_ATTEMPTS):
        if len(seeds) >= n_regions:
            break
        seeds.add(random.choice(free_cells))
    if len(seeds) < n_regions:
        return None

    frontier: list[tuple[int, int, int]] = []
    for rid, (r, c) in enumerate(list(seeds)):
        region[r, c] = rid
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and region[nr, nc] == -1:
                frontier.append((nr, nc, rid))
    random.shuffle(frontier)

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if region[r, c] != -1 or not mask[r, c]:
            continue
        region[r, c] = rid
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and region[nr, nc] == -1:
                frontier.append((nr, nc, rid))

    if np.any(mask & (region < 0)):
        return None
    for rid in range(n_regions):
        comp_mask = region == rid
        if int(comp_mask.sum()) < MIN_REGION_CELLS:
            return None
        _, n = label_mask(comp_mask, connectivity=4)
        if int(n) != 1:
            return None
    return region


def _build_nested_layout(H: int, W: int, n_regions: int) -> np.ndarray | None:
    outside_n = n_regions - 2
    margin = 2
    for _ in range(LAYOUT_RETRIES):
        top_t = random.randint(4, 5)
        bottom_t = random.randint(4, 5)
        left_t = random.randint(4, 5)
        right_t = random.randint(4, 5)

        max_inner_h = min(8, H - 2 * margin - top_t - bottom_t)
        max_inner_w = min(8, W - 2 * margin - left_t - right_t)
        if max_inner_h < 4 or max_inner_w < 4:
            continue
        inner_h = random.randint(4, max_inner_h)
        inner_w = random.randint(4, max_inner_w)
        outer_h = top_t + inner_h + bottom_t
        outer_w = left_t + inner_w + right_t

        if outer_h > H - 2 * margin or outer_w > W - 2 * margin:
            continue

        r0 = random.randint(margin, H - margin - outer_h)
        c0 = random.randint(margin, W - margin - outer_w)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        ir0 = r0 + top_t
        ic0 = c0 + left_t
        ir1 = ir0 + inner_h - 1
        ic1 = ic0 + inner_w - 1

        region = np.full((H, W), -1, dtype=int)
        region[r0 : r1 + 1, c0 : c1 + 1] = 0
        region[ir0 : ir1 + 1, ic0 : ic1 + 1] = 1

        free_mask = region < 0
        outside = _grow_regions_on_mask(free_mask, outside_n)
        if outside is None:
            continue
        region[free_mask] = outside[free_mask] + 2

        assert int((region == 1).sum()) >= MIN_REGION_CELLS
        assert int((region == 0).sum()) >= MIN_REGION_CELLS
        return region
    return None


def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    H, W = region.shape
    undirected = set()
    for r in range(H):
        for c in range(W):
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if _in_bounds(region.shape, nr, nc):
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
    return n_markers, n_normal, n_colored


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
    *,
    min_source_steps: int = 0,
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
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == int(src)
                    for bd in bar
                ):
                    continue

                step = 2
                source_steps = 0
                found = False
                while True:
                    nr, nc = r + stem[0] * step, c + stem[1] * step
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied_cells or (nr, nc) in cells:
                        step += 1
                        continue
                    rid = int(region[nr, nc])
                    if rid == int(src):
                        source_steps += 1
                        step += 1
                        continue
                    found = rid == int(dst) and source_steps >= int(min_source_steps)
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


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[tuple]) -> None:
    for kind, center, stem, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = int(tc)
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[tuple], out_colors: list[int]) -> None:
    for _kind, center, stem, src, _dst, _tc in placed_markers:
        color = int(out_colors[src])
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[tuple],
    occupied_neighbors: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
) -> str | None:
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
        output_grid[lr + dr, lc + dc] = int(out_colors[rid])
    return orient


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    H = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    W = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)

    region = _build_nested_layout(H, W, n_regions)
    if region is None:
        return None

    directed_pairs = _find_adjacent_pairs(region)
    if (0, 1) not in directed_pairs:
        return None

    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    n_markers, _n_normal, n_colored = counts

    marker_pairs: list[tuple[int, int]] = [(0, 1)]
    dest_used = {1}
    remaining_pairs = [(src, dst) for src, dst in directed_pairs if (src, dst) != (0, 1)]
    random.shuffle(remaining_pairs)
    for src, dst in remaining_pairs:
        if dst in dest_used:
            continue
        marker_pairs.append((src, dst))
        dest_used.add(dst)
        if len(marker_pairs) >= n_markers:
            break
    if len(marker_pairs) < n_markers:
        n_markers = len(marker_pairs)
        if n_markers < MARKER_MIN:
            return None
        n_colored = min(n_colored, n_markers - 2)

    other_pairs = marker_pairs[1:]
    random.shuffle(other_pairs)
    colored_pairs = set(other_pairs[:n_colored])
    normal_pairs = [pair for pair in marker_pairs if pair not in colored_pairs]

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]
    out_colors = list(input_colors)

    colored_targets: dict[tuple[int, int], int] = {}
    for src, dst in colored_pairs:
        cands = [c for c in pool if c != input_colors[dst]]
        if not cands:
            return None
        colored_targets[(src, dst)] = random.choice(cands)

    for src, dst in normal_pairs:
        out_colors[dst] = input_colors[src]
    for pair, tc in colored_targets.items():
        out_colors[pair[1]] = int(tc)

    marker_defs = [
        {"kind": "normal", "src": 0, "dst": 1, "tc": None, "min_source_steps": 2},
    ]
    for src, dst in other_pairs:
        marker_defs.append(
            {
                "kind": "colored" if (src, dst) in colored_pairs else "normal",
                "src": src,
                "dst": dst,
                "tc": colored_targets.get((src, dst)),
                "min_source_steps": 0,
            }
        )

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers: list[tuple] = []
    pending = marker_defs[:]
    while pending:
        scored = []
        for idx, md in enumerate(pending):
            cands = _find_valid_placements(
                region,
                md["src"],
                md["dst"],
                occupied_cells,
                occupied_neighbors,
                H,
                W,
                min_source_steps=int(md["min_source_steps"]),
            )
            if not cands:
                return None
            priority = 0 if int(md["min_source_steps"]) > 0 else 1
            scored.append((priority, len(cands), idx, cands))
        priority, _count, pick_idx, cands = min(scored, key=lambda t: (t[0], t[1], t[2]))
        md = pending.pop(pick_idx)
        center, stem = random.choice(cands)
        _register_marker(md["kind"], center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((md["kind"], center, stem, md["src"], md["dst"], md["tc"]))

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
