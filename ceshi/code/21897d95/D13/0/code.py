# Variant of 21897d95
# Dimension: D13 — T标记放置位置约束 (meta)
# Change: T标记放置在两个区域的边界上，茎跨越边界指向目标区域，对角线单元在源区域中... → T标记完全在源区域内部
#
# Design analysis for the D13 variant
# - Concrete behavior:
#   The visible control object is still the same 4-cell T (or colored-center T), but every
#   marker cell now lies inside the source region instead of straddling a source/destination
#   boundary. The stem still indicates the direction of influence, but the solver must now
#   travel through some amount of source-colored interior before it reaches the nearest
#   boundary in that direction and then the destination region beyond it.
# - Behavioral trace:
#   Original: for a downward T on a boundary, the solver could take the two diagonals at
#   center + stem + bar as source evidence and could find the destination by scanning from
#   the center starting at step 2, because the stem already touched the boundary.
#   Variant: the same downward T is fully surrounded by source cells. The source is the
#   owning region around the T, and the destination is the first different region reached
#   only after skipping the T cells and the intervening source interior along the stem ray.
#   The difference is exactly where source/destination are inferred from: local boundary
#   geometry is replaced by owner-region inference plus a nearest-boundary ray.
# - Complexity calibration:
#   To make the change visible and non-degenerate, the generator requires the stem direction
#   to be the unique nearest boundary direction from the T center, and it keeps the marker at
#   least one full source cell away from that boundary. That forces a measurable gap and
#   prevents samples from collapsing back into the boundary-touching original behavior.
# - Implicit assumptions audited from the original solver:
#   The original `first_component_from(..., start=2)` assumed the destination starts almost
#   immediately after the T. That is false now, so destination search is rewritten to skip
#   marker holes and same-source fragments until the first foreign component.
#   The original source lookup via `center + stem + bar` relied on the T lying on a boundary.
#   That is also false now, so source ownership is inferred from the components adjacent to
#   the marker cavity.
#   The old small-scrap heuristic becomes unsafe here because an internal T naturally splits
#   its owner region into several small fragments; those fragments can be tiny and
#   non-rectangular even though they are not scraps. The generator never created scraps in the
#   original code, so this variant drops that heuristic to avoid false positives.
# - Placement legality:
#   Generator placement checks guarantee that all T cells and their 4-neighborhood ring stay
#   inside the source region, that markers are buffered from each other, that the first
#   foreign region reached along the stem is the intended destination, and that the stem
#   direction is the unique nearest boundary direction from the center.
# - Visual observability:
#   A human can now see the T floating inside one monochrome area with a clear corridor from
#   the stem toward the closest border. Across examples the repeated pattern is that the stem
#   points toward the nearest edge of its containing region, not to an immediate boundary
#   crossing under the marker.
# - Visual cleanliness and color inference:
#   Region colors remain distinct in the input, recoloring rules stay unchanged, and markers
#   are kept separated by buffer cells so the source region, T marker, and optional L9 marker
#   remain easy to parse.

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

    def attach_owner(marker: dict) -> None:
        counts = adjacent_component_counts(marker["cells"])
        assert counts, "internal T marker must touch its owner region"
        owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
        owner_color = int(comp_colors[owner])
        assert all(
            int(comp_colors[comp]) == owner_color for comp in counts
        ), "internal T marker must be fully embedded in one source color"
        for comp in counts:
            dsu.union(owner, comp)
        marker["source_comp"] = owner

    for marker in markers:
        attach_owner(marker)

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    def first_foreign_component_from(
        center: tuple[int, int], direction: tuple[int, int], source_root: int
    ) -> tuple[int, int] | None:
        r, c = center
        step = 1
        seen_source = False
        while True:
            rr = r + direction[0] * step
            cc = c + direction[1] * step
            if not _in_bounds(g.shape, rr, cc):
                return None
            cid = int(comp_id[rr, cc])
            if cid < 0:
                step += 1
                continue
            root = dsu.find(cid)
            if root == source_root:
                seen_source = True
                step += 1
                continue
            assert seen_source, "stem must travel through its owner region before leaving it"
            return cid, step

    final_color = dict(root_color)
    for marker in markers:
        source_root = dsu.find(marker["source_comp"])
        marker["source"] = source_root

        exit_distances: dict[tuple[int, int], int] = {}
        for direction in DIRS:
            result = first_foreign_component_from(marker["center"], direction, source_root)
            if result is not None:
                _cid, steps = result
                exit_distances[direction] = int(steps)

        stem = marker["stem"]
        assert stem in exit_distances, "T stem must eventually leave the owner region"
        stem_dist = exit_distances[stem]
        assert all(
            exit_distances[d] > stem_dist for d in exit_distances if d != stem
        ), "stem must point to the unique nearest boundary direction"

        dest_result = first_foreign_component_from(marker["center"], stem, source_root)
        assert dest_result is not None
        dest_comp, _ = dest_result
        dest = dsu.find(dest_comp)
        replacement = (
            int(marker["target_color"]) if marker["kind"] == "colored" else int(root_color[source_root])
        )
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
    Repaint regions from T-shaped control markers that sit inside their source regions, then optionally rotate via a color-9 L-marker.

    Input structure:
    - The grid is a mosaic of monochrome regions.
    - Color 1 still draws T-shaped control markers, and some controls still use a colored center with three singleton color-1 branches.
    - Unlike the original task, every T marker is fully embedded inside its source region instead of sitting on a boundary.
    - An optional 5-cell L-shaped color-9 marker triggers a 90-degree rotation of the final output.

    Rule:
    - The T stem shows which direction to look for the nearest boundary from the marker center.
    - The region surrounding the T is the source region; following the stem through that source interior until the first different region identifies the destination.
    - A normal T repaints the destination to the source region's original color, while a colored-center T repaints it to the center color.
    - Every marker cell is filled with the final color of its surrounding source region.
    - A 5-cell L-shaped color-9 marker controls rotation: ┐ (corner top-right) → CCW 90°, ┌ (corner top-left) → CW 90°.

    Key insight:
    - The T no longer tells you the source via boundary contact; instead, you must treat the T as living inside one region and shoot the stem toward that region's nearest boundary to find what changes.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (internal-T + l9rot) ---------------

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

# Variant-specific geometry
MIN_STEM_EXIT = 3


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
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _ray_exit(
    region: np.ndarray, center: tuple[int, int], direction: tuple[int, int], src: int
) -> tuple[int, int] | None:
    r, c = center
    step = 1
    while True:
        nr = r + direction[0] * step
        nc = c + direction[1] * step
        if not (0 <= nr < region.shape[0] and 0 <= nc < region.shape[1]):
            return None
        hit = int(region[nr, nc])
        if hit != src:
            return step, hit
        step += 1


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
        for dr, dc in DIRS:
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
        for dr, dc in DIRS:
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
            for dr, dc in DIRS:
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


def _corridor_clear(
    center: tuple[int, int],
    stem: tuple[int, int],
    stop_step: int,
    cells: set[tuple[int, int]],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> bool:
    for step in range(1, stop_step + 1):
        pos = (center[0] + stem[0] * step, center[1] + stem[1] * step)
        if pos in cells:
            continue
        if pos in occupied_cells or pos in occupied_neighbors:
            return False
    return True


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
            if int(region[r, c]) != src:
                continue
            center = (r, c)
            for stem in DIRS:
                cells = _t_cells(center, stem)
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                if not all(int(region[cr, cc]) == src for cr, cc in cells):
                    continue

                neighborhood = _t_neighborhood(center, stem)
                if not all(
                    0 <= nr < H and 0 <= nc < W and int(region[nr, nc]) == src
                    for nr, nc in neighborhood
                ):
                    continue

                if cells & occupied_cells:
                    continue
                if neighborhood & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue

                stem_exit = _ray_exit(region, center, stem, src)
                if stem_exit is None:
                    continue
                stem_dist, stem_hit = stem_exit
                if stem_hit != dst or stem_dist < MIN_STEM_EXIT:
                    continue

                unique_nearest = True
                for other in DIRS:
                    if other == stem:
                        continue
                    other_exit = _ray_exit(region, center, other, src)
                    if other_exit is not None and other_exit[0] <= stem_dist:
                        unique_nearest = False
                        break
                if not unique_nearest:
                    continue

                if not _corridor_clear(center, stem, stem_dist, cells, occupied_cells, occupied_neighbors):
                    continue

                candidates.append((center, stem))
    return candidates


def _register_marker(
    center: tuple[int, int],
    stem: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
    cells = _t_cells(center, stem)
    occupied_cells |= cells
    occupied_neighbors |= (_t_neighborhood(center, stem) - cells)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[tuple]) -> None:
    for kind, center, stem, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = tc
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[tuple], out_colors: list[int]) -> None:
    for _kind, center, stem, src, _dst, _tc in placed_markers:
        color = out_colors[src]
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
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, stem, *_rest in placed_markers:
        all_marker_cells |= _t_cells(center, stem)

    orientations = list(L9_OFFSETS.items())
    random.shuffle(orientations)
    for orient, offsets in orientations:
        candidates = []
        for r in range(H):
            for c in range(W):
                cells = [(r + dr, c + dc) for dr, dc in offsets]
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                rid = int(region[r, c])
                if int(input_grid[r, c]) == 9:
                    continue
                if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                    continue
                if any((cr, cc) in all_marker_cells for cr, cc in cells):
                    continue
                if any((cr, cc) in occupied_neighbors for cr, cc in cells):
                    continue
                candidates.append((r, c, rid))
        if not candidates:
            continue
        lr, lc, rid = random.choice(candidates)
        for dr, dc in offsets:
            input_grid[lr + dr, lc + dc] = 9
            output_grid[lr + dr, lc + dc] = out_colors[rid]
        return orient
    return None


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
        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, tc))

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
