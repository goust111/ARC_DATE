# Combo variant of 21897d95 (index 14)
# combo_spec: D7/1+D12/0
# Name: 计数全覆盖旋转

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import label_mask
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import label_mask


# Design analysis for the combo variant:
# - Concrete appearance and behavior trace:
#   The board is still an irregular region mosaic with T controls on boundaries. Normal Ts are
#   four color-1 cells; colored Ts have a literal colored center plus three singleton color-1
#   branches. The old color-9 L rotation cue is removed. Instead, every region is targeted by
#   exactly one T, so the total T count equals the visible region count. Example: with six
#   regions, the generator places six Ts, each stem choosing one distinct destination region.
#   Solver flow becomes: parse all Ts -> recolor every region exactly once -> count Ts -> rotate
#   the finished board by that count (4 none, 5 CW, 6 180, 7 CCW). Compared with the source
#   task, rotation is no longer attached to a separate object; it is encoded by the same T
#   population that already drives recoloring.
# - Complexity calibration:
#   The new signal has four states in generated data because region count is restricted to 4..7.
#   This avoids the weak 8->no-rotation wraparound case from the raw modulo rule, so the count
#   cue stays visually sharp while remaining small enough for humans to infer from 2-4 examples.
#   Full coverage also makes the difference from the original task obvious: no region remains
#   untouched after solving.
# - Hidden assumptions audited:
#   Original `main()` assumed rotation could be decided after recoloring by reading an optional
#   L9 object. That assumption is deleted: `_solve_unrotated()` now returns both the board and
#   the number of parsed T markers. Original conflict checking only prevented incompatible writes
#   to one destination; in this combo we must also assert uniqueness and completeness of targets.
#   The loops that scan colored-center Ts still rely on each branch color-1 cell being a
#   singleton not already consumed by a tetromino; generator placement preserves that invariant.
#   The destination lookup still assumes the first non-control component found two or more cells
#   along the stem is the target region; generator only accepts placements with that geometry.
# - Placement legality:
#   Marker placement uses backtracking over destinations. Each candidate T must stay in bounds,
#   avoid all occupied cells, keep colored branches singleton by buffering neighbors, place the
#   two source diagonals in the source region, and see the destination region first along the
#   stem ray. This guarantees non-overlap and that solver parsing matches generation.
# - Visual observability:
#   A human can read the rule from examples because the same visible quantity, the number of T
#   markers, changes the global orientation. The generator rejects non-zero-rotation cases whose
#   recolored board is rotationally invariant, so the count cue actually changes the output.
#   Full coverage gives an additional checksum: every region changes, not just a sparse subset.
# - Visual cleanliness:
#   No L marker and no extra scraps are generated. That keeps the scene to regions plus Ts only,
#   which matters because full coverage already increases marker density. Larger minimum canvases
#   for 6-7 regions preserve spacing between controls.
# - Change inferability:
#   Region colors are distinct in the input, so normal Ts imply a readable donor->receiver color
#   transfer. Colored-center Ts still expose literal replacement colors. Because all destinations
#   are structurally marked and the rotation state depends only on a small visible count, the
#   transformation remains inferable without hidden ordering by color id or any off-grid cue.
# - New main() assumptions and generator support:
#   `main()` now assumes exactly one T per destination root and a total of 4..8 Ts. Generator
#   guarantees one target per region, distinct region colors, and in practice samples 4..7
#   regions so each sample has one of four unambiguous rotation states.


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


def _rotation_k_from_t_count(t_marker_count: int) -> int:
    assert 4 <= int(t_marker_count) <= 8, "expected 4-8 T markers"
    clockwise_turns = (int(t_marker_count) - 4) % 4
    return (-clockwise_turns) % 4


def _solve_unrotated(g: np.ndarray) -> tuple[np.ndarray, int]:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        k: {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == k)}
        for k in range(1, int(n_1) + 1)
    }
    one_sizes = {k: len(cells) for k, cells in one_components.items()}

    markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)
    t_marker_count = 0

    for cells in one_components.values():
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        markers.append({"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells})
        t_marker_count += 1
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
                t_marker_count += 1
                used_ones |= set(cells)
                for rr, cc in marker_cells:
                    control[rr, cc] = True

    assert 4 <= t_marker_count <= 8, "rotation encoding expects 4-8 T markers"

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a T control marker"

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

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    final_color = dict(root_color)
    target_counts: Counter = Counter()
    for marker in markers:
        dest = dsu.find(marker["dest_comp"])
        source = dsu.find(marker["source_comp"])
        marker["source"] = source
        assert target_counts[dest] == 0, "each region must be targeted by exactly one T marker"
        target_counts[dest] += 1
        replacement = int(marker["target_color"]) if marker["kind"] == "colored" else int(root_color[source])
        final_color[dest] = replacement

    assert set(target_counts) == set(root_color), "every region must be targeted by exactly one T marker"

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
    return out, t_marker_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint every region from T-shaped boundary controls, erase those controls, and finally rotate by the total T count.

    Input structure:
    - The grid is partitioned into irregular monochrome regions.
    - Color 1 forms T markers on region boundaries; some markers instead use a colored center with three singleton color-1 branches.
    - There is no separate L-shaped rotation sprite.
    - Every base region is the destination of exactly one T marker, so the T population fully covers the region set.

    Rule:
    - For a normal T, the single stem points into the destination region, while the two diagonal cells behind that stem identify the source-side region.
    - The destination takes the source region's original color; for a colored-center T, it takes the center color instead.
    - Every marker cell is filled with the final color of the source-side region.
    - After all recoloring is resolved simultaneously, rotate the whole board by the total T count: 4 -> no rotation, 5 -> clockwise 90, 6 -> 180, 7 -> counterclockwise 90.

    Key insight:
    - The same T markers carry both pieces of information: locally they say which region donates color to which target, and globally their total count encodes the final orientation.
    """
    g = np.asarray(grid, dtype=int)
    out, t_marker_count = _solve_unrotated(g)
    rot_k = _rotation_k_from_t_count(t_marker_count)
    return np.rot90(out, rot_k) if rot_k else out


# --------------- generator (count full-cover rotation) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 3.0
REGION_COUNTS = [4, 5, 6, 7]

NORMAL_MAX = 6
COLORED_MAX = 2

REGION_COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8, 9]
REPLACEMENT_COLOR_POOL = [c for c in range(10) if c != 1]

SEED_ATTEMPTS = 200
GEN_RETRIES = 300

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_region_count_cycle: list[int] = []
_region_count_index = 0


def _next_region_count() -> int:
    global _region_count_cycle, _region_count_index
    if _region_count_index >= len(_region_count_cycle):
        _region_count_cycle = list(REGION_COUNTS)
        random.shuffle(_region_count_cycle)
        _region_count_index = 0
    n_regions = _region_count_cycle[_region_count_index]
    _region_count_index += 1
    return int(n_regions)


def _choose_marker_counts(n_regions: int) -> tuple[int, int, int] | None:
    min_colored = max(0, int(n_regions) - NORMAL_MAX)
    max_colored = min(COLORED_MAX, int(n_regions) - 2)
    if min_colored > max_colored:
        return None
    n_colored = random.randint(min_colored, max_colored)
    n_normal = int(n_regions) - n_colored
    return int(n_regions), int(n_normal), int(n_colored)


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
                    a = int(region[r, c])
                    b = int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _build_sources_by_dest(n_regions: int, directed_pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    by_dest = {dst: [] for dst in range(n_regions)}
    for src, dst in directed_pairs:
        by_dest[dst].append(src)
    for dst in by_dest:
        random.shuffle(by_dest[dst])
    return by_dest


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


def _place_full_cover_markers(
    region: np.ndarray,
    marker_defs: dict[int, tuple[str, int | None]],
    sources_by_dest: dict[int, list[int]],
    H: int,
    W: int,
) -> list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]] | None:
    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]] = []
    remaining = set(marker_defs)

    def backtrack() -> bool:
        if not remaining:
            return True

        best_dst = -1
        best_options: list[tuple[int, tuple[int, int], tuple[int, int]]] | None = None
        dst_order = list(remaining)
        random.shuffle(dst_order)

        for dst in dst_order:
            options: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
            src_order = list(sources_by_dest[dst])
            random.shuffle(src_order)
            for src in src_order:
                cands = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W)
                if not cands:
                    continue
                random.shuffle(cands)
                options.extend((src, center, stem) for center, stem in cands)
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_dst = dst
                best_options = options
                if len(best_options) == 1:
                    break

        assert best_options is not None and best_dst >= 0
        kind, target_color = marker_defs[best_dst]
        random.shuffle(best_options)
        for src, center, stem in best_options:
            prev_cells = occupied_cells.copy()
            prev_neighbors = occupied_neighbors.copy()
            _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
            placed.append((kind, center, stem, src, best_dst, target_color))
            remaining.remove(best_dst)
            if backtrack():
                return True
            remaining.add(best_dst)
            placed.pop()
            occupied_cells.clear()
            occupied_cells.update(prev_cells)
            occupied_neighbors.clear()
            occupied_neighbors.update(prev_neighbors)
        return False

    return placed if backtrack() else None


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
) -> None:
    for kind, center, stem, _src, _dst, target_color in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = 1
        else:
            assert target_color is not None
            grid[center[0], center[1]] = int(target_color)
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
    out_colors: list[int],
) -> None:
    for _kind, center, stem, src, _dst, _target_color in placed_markers:
        color = out_colors[src]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _gen_attempt(n_regions: int) -> np.ndarray | None:
    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    n_markers, _n_normal, n_colored = counts
    assert n_markers == n_regions

    min_size = max(11, int(n_regions * REGION_SIZE_FACTOR + 0.999))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    directed_pairs = _find_adjacent_pairs(region)
    sources_by_dest = _build_sources_by_dest(n_regions, directed_pairs)
    if any(not sources_by_dest[dst] for dst in range(n_regions)):
        return None

    pool = list(REGION_COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    colored_dests = set(random.sample(range(n_regions), n_colored))
    marker_defs: dict[int, tuple[str, int | None]] = {}
    for dst in range(n_regions):
        if dst in colored_dests:
            cands = [c for c in REPLACEMENT_COLOR_POOL if c != input_colors[dst]]
            if not cands:
                return None
            marker_defs[dst] = ("colored", random.choice(cands))
        else:
            marker_defs[dst] = ("normal", None)

    placed_markers = _place_full_cover_markers(region, marker_defs, sources_by_dest, H, W)
    if placed_markers is None or len(placed_markers) != n_regions:
        return None

    out_colors = list(input_colors)
    for kind, _center, _stem, src, dst, target_color in placed_markers:
        if kind == "colored":
            assert target_color is not None
            out_colors[dst] = int(target_color)
        else:
            out_colors[dst] = input_colors[src]

    if not all(out_colors[rid] != input_colors[rid] for rid in range(n_regions)):
        return None

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)
    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    rot_k = _rotation_k_from_t_count(n_markers)
    if rot_k:
        rotated_output = np.rot90(output_grid, rot_k)
        if np.array_equal(rotated_output, output_grid):
            return None
        output_grid = rotated_output

    try:
        result = main(input_grid)
    except Exception:
        return None
    return input_grid if np.array_equal(result, output_grid) else None


def generate_input() -> np.ndarray:
    n_regions = _next_region_count()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(n_regions)
        if result is not None:
            return result
    raise RuntimeError(
        f"Failed to generate valid input with {n_regions} regions after {GEN_RETRIES} attempts"
    )
