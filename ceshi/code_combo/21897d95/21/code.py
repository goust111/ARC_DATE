# Combo variant of 21897d95 (index 21)
# combo_spec: D3/2+D7/2
# Name: 箭头标记与镜像
#
# Design analysis for this combo variant
# - Concrete fused behavior:
#   The old 4-cell T is replaced by a 5-cell arrow. We keep the original logical anchor:
#   the "center" is still the middle cell whose forward direction selects the destination,
#   but the marker now adds one tail cell behind that center. For stem RIGHT, the arrow cells are
#   {(r,c-1), (r,c), (r,c+1), (r-1,c+1), (r+1,c+1)}. The destination is still the first
#   base region reached from (r,c) along RIGHT starting at step 2, while the source is now the
#   region the arrow comes out of, read from the tail-side ownership around (r,c-1). The visual
#   difference from the
#   original T is clear: the marker now looks like ">" instead of a T while keeping the same
#   recoloring semantics. The optional L9 marker no longer rotates the solved mosaic; it flips
#   it horizontally for ┐ and vertically for ┌, and that mirror is applied only after all
#   arrow-driven recoloring is complete.
# - Small traced example:
#   Suppose a RIGHT-pointing control sits between a blue source region on the left and a green
#   destination region on the right. In the original task, a T with stem RIGHT recolors green
#   using blue. Here the arrow ">" does the same recolor first. If the grid also contains a
#   top-right L9, the finished recolored mosaic is then mirrored left-right. After that mirror,
#   the erased arrow-colored patch appears on the opposite side, so a human must not read the
#   output as if the visible post-flip orientation had been the original cue.
# - Complexity calibration:
#   The arrow keeps only four axial directions, so the rule family does not explode, but the
#   recognition task is meaningfully different because the source cue is now embedded in a
#   5-cell silhouette rather than a 4-cell T. The generator keeps multiple arrows per board and
#   rejects mirror cases that are visually inert or accidentally collapse back to the legacy
#   rotation behavior, so the fusion effect stays observable.
# - Hidden-assumption review:
#   1. The old normal-marker parser assumed every color-1 control was a single 4-cell component.
#      That no longer holds for colored arrows: the color-1 part is a 3-cell head cluster plus
#      one singleton tail around a colored center. The solver therefore parses normal arrows from
#      5-cell color-1 components and colored arrows by local geometry around each non-1 center.
#   2. The old source lookup from off-marker diagonal cells no longer applies, because the arrow's
#      diagonal head cells are themselves part of the control shape. The solver instead infers the
#      source from the arrow tail: the marker emerges from that region and points across the border.
#   3. Small irregular scraps and the L9 marker are still absorbed as control-colored regions
#      after explicit arrow cells are masked out. This remains safe because arrow cells are marked
#      in `control` before the small-component pass.
#   4. Mirror handling must stay outside the recoloring solver. `_solve_pre_mirror()` consumes the
#      unflipped arrows, while `main()` applies the L9-triggered flip only afterward.
# - Placement legality:
#   The generator reserves every arrow cell plus a neighbor buffer, so normal arrows stay isolated
#   5-cell color-1 components. Colored arrows get a stronger buffer: the tail must remain a 1-cell
#   singleton and the front+two head cells must remain a 3-cell color-1 component, because the
#   solver uses those exact component sizes to recognize the colored-arrow pattern.
# - Visual observability / cleanliness:
#   Arrows are larger than the old T markers, so the generator slightly favors roomier boards and
#   keeps the same "distinct destinations" policy to avoid overlapping recolor instructions. L9 is
#   only placed as a visible standalone 9-shape inside a non-9 visible region, so the mirror cue
#   is easy to isolate by eye.
# - Fusion-specific visibility check:
#   For L9 cases, the generator rejects outputs where the mirrored result equals the unmirrored
#   solved mosaic or matches the legacy rotation target. This ensures the mirror trigger is not a
#   no-op and does not visually collapse back to the old task family.

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


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if int(stem[0]) == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _arrow_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = map(int, center)
    dr, dc = map(int, stem)
    bar = _get_bar(stem)
    front = (r + dr, c + dc)
    return {
        (r - dr, c - dc),
        (r, c),
        front,
        (front[0] + bar[0][0], front[1] + bar[0][1]),
        (front[0] + bar[1][0], front[1] + bar[1][1]),
    }


def _arrow_branch_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    return _arrow_cells(center, stem) - {tuple(map(int, center))}


def _arrow_source_diagonals(center: tuple[int, int], stem: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, center)
    dr, dc = map(int, stem)
    bar = _get_bar(stem)
    front = (r + dr, c + dc)
    return [(front[0] + bd[0], front[1] + bd[1]) for bd in bar]


def _arrow_tail(center: tuple[int, int], stem: tuple[int, int]) -> tuple[int, int]:
    return (int(center[0]) - int(stem[0]), int(center[1]) - int(stem[1]))


def _arrow_front_cluster(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = map(int, center)
    dr, dc = map(int, stem)
    front = (r + dr, c + dc)
    return {front, *_arrow_source_diagonals(center, stem)}


def _arrow_neighborhood(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _arrow_cells(center, stem)
    neighborhood: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (int(r) + int(dr), int(c) + int(dc))
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _parse_arrow_pentomino(
    cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    if len(cells) != 5:
        return None
    for r, c in cells:
        center = (int(r), int(c))
        for stem in DIRS:
            if cells == _arrow_cells(center, stem):
                return center, stem, _get_bar(stem)
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


def _solve_pre_mirror(g: np.ndarray) -> np.ndarray:
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
        parsed = _parse_arrow_pentomino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        markers.append({"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells})
        used_ones |= cells
        for r, c in cells:
            control[int(r), int(c)] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            matches: list[tuple[tuple[int, int], list[tuple[int, int]], set[tuple[int, int]]]] = []
            center = (int(r), int(c))
            for stem in DIRS:
                branch = _arrow_branch_cells(center, stem)
                if not all(_in_bounds(g.shape, rr, cc) for rr, cc in branch):
                    continue
                if any(int(g[rr, cc]) != 1 for rr, cc in branch):
                    continue
                if branch & used_ones:
                    continue
                tail = _arrow_tail(center, stem)
                front_cluster = _arrow_front_cluster(center, stem)
                front = (int(center[0]) + int(stem[0]), int(center[1]) + int(stem[1]))
                tail_label = int(labels_1[tail])
                front_label = int(labels_1[front])
                if tail_label == 0 or front_label == 0:
                    continue
                if one_sizes[tail_label] != 1 or one_sizes[front_label] != 3:
                    continue
                if any(int(labels_1[rr, cc]) != front_label for rr, cc in front_cluster):
                    continue
                if tail_label == front_label:
                    continue
                side_ok = True
                for side in _get_bar(stem):
                    rr, cc = int(r + side[0]), int(c + side[1])
                    if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == 1:
                        side_ok = False
                        break
                if not side_ok:
                    continue
                matches.append((stem, _get_bar(stem), branch))
            assert len(matches) <= 1, "colored arrow center must have a unique orientation"
            if not matches:
                continue
            stem, bar, branch = matches[0]
            marker_cells = set(branch) | {center}
            markers.append(
                {
                    "kind": "colored",
                    "center": center,
                    "stem": stem,
                    "bar": bar,
                    "cells": marker_cells,
                    "target_color": int(g[r, c]),
                }
            )
            used_ones |= branch
            for rr, cc in marker_cells:
                control[int(rr), int(cc)] = True

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to an arrow control marker"

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
                control[int(rr), int(cc)] = True

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"
    dsu = _DSU(len(comp_colors))

    def first_component_from(center: tuple[int, int], direction: tuple[int, int], start: int) -> int | None:
        r, c = center
        step = int(start)
        while True:
            rr = int(r + direction[0] * step)
            cc = int(c + direction[1] * step)
            if not _in_bounds(g.shape, rr, cc):
                return None
            if comp_id[rr, cc] >= 0:
                return int(comp_id[rr, cc])
            step += 1

    def adjacent_component_counts(cells: set[tuple[int, int]]) -> Counter:
        counts: Counter = Counter()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = int(r + dr), int(c + dc)
                if _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0:
                    counts[int(comp_id[rr, cc])] += 1
        return counts

    for marker in markers:
        if marker["kind"] in {"normal", "colored"}:
            center = marker["center"]
            stem = marker["stem"]
            dest = first_component_from(center, stem, start=2)
            assert dest is not None, "arrow marker must point at a destination region"

            tail_counts = adjacent_component_counts({_arrow_tail(center, stem)})
            assert tail_counts, "arrow tail must touch its source-side region"
            source = max(tail_counts, key=lambda comp: (tail_counts[comp], comp_areas[comp]))

            for comp in adjacent_component_counts(marker["cells"]):
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
            marker["dest_comp"] = dest
            marker["source_comp"] = source
        else:
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "small marker must touch its owner region"
            owner = max(counts, key=lambda comp: (counts[comp], comp_areas[comp]))
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
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting arrow controls for one region"
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
            out[int(r), int(c)] = color

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
                assert False, f"unobserved L9 corner at ({cr},{cc})"
    return None


def _apply_l9_transform(grid: np.ndarray, orient: str | None) -> np.ndarray:
    if orient == "top_right":
        return np.fliplr(grid)
    if orient == "top_left":
        return np.flipud(grid)
    return grid


def _apply_legacy_rotation(grid: np.ndarray, orient: str | None) -> np.ndarray:
    if orient == "top_right":
        return np.rot90(grid, 1)
    if orient == "top_left":
        return np.rot90(grid, 3)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor mosaic regions from arrow-shaped boundary controls, erase those controls, and optionally mirror the finished mosaic from a color-9 L marker.

    Input structure:
    - The grid is partitioned into monochrome regions with irregular borders.
    - Color 1 now forms 5-cell arrows instead of T tetrominoes: a 3-cell shaft plus a 2-cell diagonal arrowhead.
    - Some arrows use a colored middle shaft cell with four surrounding color-1 arrow cells; that middle color is a literal replacement color.
    - An optional 5-cell color-9 L marker sits inside one region and acts only as a whole-grid post-processing trigger.

    Rule:
    - For each arrow, the shaft direction points toward the destination region; read that destination from the first base region reached two steps ahead of the arrow's middle cell.
    - The source side is the region the arrow emerges from at its tail; a plain color-1 arrow copies that source region's original color, while a colored-center arrow uses its center color instead.
    - After recoloring, every arrow cell is filled with the final color of the source-side region.
    - If a color-9 L is present, mirror the already-solved mosaic afterward: ┐ means a horizontal flip and ┌ means a vertical flip.

    Key insight:
    - The arrow tells you what changes before any global transform happens, while the L only flips the completed answer; if you mentally flip first, the arrow directions become misleading.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_pre_mirror(g)
    return _apply_l9_transform(out, _find_l9_orientation(g))


# --------------- generator (arrowflip) ---------------

# ── Tunable constants ──────────────────────────────────────────────────
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 3.0
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

COLOR_POOL = [c for c in range(10) if c not in {1, 9}]
COLORED_CENTER_POOL = [c for c in range(10) if c not in {1, 9}]

L9_PROBABILITY = 0.7
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 400


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    region = np.full((H, W), -1, dtype=int)
    seeds = set()
    for _ in range(SEED_ATTEMPTS):
        if len(seeds) >= n_regions:
            break
        seeds.add((random.randint(1, H - 2), random.randint(1, W - 2)))
    if len(seeds) < n_regions:
        return None

    frontier: list[tuple[int, int, int]] = []
    for rid, (r, c) in enumerate(list(seeds)):
        region[r, c] = rid
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and region[nr, nc] == -1:
                frontier.append((nr, nc, rid))
    random.shuffle(frontier)

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if region[r, c] != -1:
            continue
        region[r, c] = rid
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and region[nr, nc] == -1:
                frontier.append((nr, nc, rid))

    for rid in range(n_regions):
        mask = region == rid
        if int(mask.sum()) < MIN_REGION_CELLS:
            return None
        _, n = label_mask(mask, connectivity=4)
        if int(n) != 1:
            return None
    return region


def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    H, W = region.shape
    undirected = set()
    for r in range(H):
        for c in range(W):
            for dr, dc in DIRS:
                nr, nc = int(r + dr), int(c + dc)
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


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            center = (int(r), int(c))
            for stem in DIRS:
                cells = _arrow_cells(center, stem)
                if not all(0 <= rr < H and 0 <= cc < W for rr, cc in cells):
                    continue
                if cells & occupied_cells:
                    continue
                if _arrow_neighborhood(center, stem) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue
                tail = _arrow_tail(center, stem)
                if int(region[tail]) != src:
                    continue
                tail_ok = True
                for dr, dc in DIRS:
                    nb = (int(tail[0] + dr), int(tail[1] + dc))
                    if nb == center:
                        continue
                    if not (0 <= nb[0] < H and 0 <= nb[1] < W and int(region[nb]) == src):
                        tail_ok = False
                        break
                if not tail_ok:
                    continue
                step = 2
                found = False
                while True:
                    nr = int(r + stem[0] * step)
                    nc = int(c + stem[1] * step)
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


def _register_marker(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
    cells = _arrow_cells(center, stem)
    occupied_cells |= cells
    if kind == "normal":
        occupied_neighbors |= (_arrow_neighborhood(center, stem) - cells)
        return

    tail = _arrow_tail(center, stem)
    front_cluster = _arrow_front_cluster(center, stem)
    for rr, cc in {tail} | front_cluster:
        allowed = {center}
        if (rr, cc) == tail:
            allowed.add(tail)
        else:
            allowed |= front_cluster
        for dr, dc in DIRS:
            nb = (int(rr + dr), int(cc + dc))
            if nb not in allowed:
                occupied_neighbors.add(nb)
    for dr, dc in DIRS:
        nb = (int(center[0] + dr), int(center[1] + dc))
        if nb not in _arrow_branch_cells(center, stem):
            occupied_neighbors.add(nb)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    return np.array(colors, dtype=int)[region]


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[tuple]) -> None:
    for kind, center, stem, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for rr, cc in _arrow_cells(center, stem):
                grid[rr, cc] = 1
        else:
            grid[int(center[0]), int(center[1])] = int(tc)
            for rr, cc in _arrow_branch_cells(center, stem):
                grid[rr, cc] = 1


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[tuple], out_colors: list[int]) -> None:
    for _kind, center, stem, src, _dst, _tc in placed_markers:
        color = int(out_colors[src])
        for rr, cc in _arrow_cells(center, stem):
            grid[rr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[tuple],
    input_grid: np.ndarray,
) -> str | None:
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, stem, *_rest in placed_markers:
        all_marker_cells |= _arrow_cells(center, stem)

    candidates: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            cells = [(int(r + dr), int(c + dc)) for dr, dc in offsets]
            if not all(0 <= rr < H and 0 <= cc < W for rr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[rr, cc]) == rid for rr, cc in cells):
                continue
            if any(int(input_grid[rr, cc]) == 9 for rr, cc in cells):
                continue
            if any((rr, cc) in all_marker_cells for rr, cc in cells):
                continue
            candidates.append((int(r), int(c), rid))

    if not candidates:
        return None
    lr, lc, _rid = random.choice(candidates)
    for dr, dc in offsets:
        rr, cc = int(lr + dr), int(lc + dc)
        input_grid[rr, cc] = 9
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


def _pre_mirror_misread_output(input_grid: np.ndarray, orient: str | None) -> np.ndarray | None:
    mirrored_input = _apply_l9_transform(input_grid, orient)
    try:
        return _solve_pre_mirror(np.asarray(mirrored_input, dtype=int))
    except Exception:
        return None


def _fusion_is_visible(input_grid: np.ndarray, solved_grid: np.ndarray, orient: str | None) -> bool:
    return _transform_is_variant_visible(solved_grid, orient)


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
        n_colored = min(n_colored, max(0, n_markers - 1))
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
        cands = [c for c in COLORED_CENTER_POOL if c != input_colors[dst]]
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
    _stamp_markers_input(input_grid, placed_markers)

    l9_orient = _try_place_l9(region, placed_markers, input_grid)
    if _find_l9_orientation(input_grid) != l9_orient:
        return None

    expected_pre = _build_color_map(region, out_colors)
    _stamp_markers_output(expected_pre, placed_markers, out_colors)
    expected = _apply_l9_transform(expected_pre, l9_orient)

    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    solved_grid = expected_pre
    if not _fusion_is_visible(input_grid, solved_grid, l9_orient):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
