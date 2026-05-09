# Variant of 21897d95
# Dimension: D7 — 旋转触发机制（L9标记） (grid_shape)
# Change: 可选5单元L形颜色9标记：┐(右上角)触发逆时针90度旋转，┌(左上角)触发顺时针90度旋转；约50%概率出现... → 旋转角度由T标记数量决定

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D7 variant
# - Concrete appearance:
#   The input still shows a mosaic of irregular monochrome regions plus 4-8 T controls.
#   Normal T markers are 4 color-1 cells. Colored T markers have a non-1 center with three
#   singleton color-1 neighbors. The old 5-cell color-9 L cue is removed entirely.
# - Behavior trace:
#   Old flow: solve recoloring from every T, then optionally rotate from a separate L9 sprite.
#   New flow: solve the same recoloring, count how many T markers exist in total, then rotate
#   from that count: 4 -> 0, 5 -> CW 90, 6 -> 180, 7 -> CCW 90, 8 -> 0.
#   Example: if a grid has 6 Ts, the recolored unrotated board is built exactly as before,
#   but the final answer is flipped by 180 instead of consulting any extra object.
# - Complexity calibration:
#   The encoded parameter has only five visible states (4-8 Ts), matching the original marker
#   count range and the source task's small modulo-based control idea. This keeps the rule
#   inferable while adding one extra reasoning step: count all T controls before rotating.
# - Hidden assumptions audited:
#   The original `main()` assumed rotation depended on a separate color-9 component and could
#   be decided after `_solve_unrotated()` without inspecting the T-marker count. In the variant,
#   that assumption breaks, so `_solve_unrotated()` now returns the number of normal+colored Ts.
#   Small scraps are still ignored for rotation because they are not T markers.
# - Placement legality:
#   The generator still uses the original non-overlap rules for Ts. Because rotation now depends
#   on count, generation chooses the marker count first and then guarantees enough regions exist
#   to host that many distinct destinations. No L9 cells are placed.
# - Visual observability:
#   For non-zero rotations, the generator rejects cases where the recolored board happens to be
#   rotationally invariant, so the count-driven rotation remains visible instead of collapsing
#   back to the unrotated appearance.
# - Cleanliness and color inference:
#   Colors still follow the original region/T semantics; no invisible ordering by color id is
#   introduced. The only new signal is the plainly visible number of T markers.


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
    return out, t_marker_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint regions from T-shaped control markers, erase the markers, and then rotate according to how many T markers appear.

    Input structure:
    - The grid is partitioned into monochrome regions.
    - Color 1 forms T-shaped control markers on region boundaries.
    - Some controls use a colored center with three singleton color-1 neighbors; that center is a literal replacement color.
    - There is no separate L-shaped rotation sprite; the total number of T markers is the only rotation cue.

    Rule:
    - For a color-1 T marker, the single stem points at the destination region to repaint.
    - The two diagonal cells behind the stem identify the source region; the destination takes that source region's original color.
    - For a colored-center T marker, the same stem identifies the destination, but the center color is copied instead.
    - All marker cells are filled with the final color of the source-side region.
    - After recoloring, rotate the whole output by the total T-marker count: 4 -> no rotation, 5 -> clockwise 90, 6 -> 180, 7 -> counterclockwise 90, 8 -> no rotation.

    Key insight:
    - Each T still tells you which region donates color and which region changes, but now the global count of those same T controls also encodes the final board rotation.
    """
    g = np.asarray(grid, dtype=int)
    out, t_marker_count = _solve_unrotated(g)
    rot_k = _rotation_k_from_t_count(t_marker_count)
    return np.rot90(out, rot_k) if rot_k else out


# --------------- generator (tcountrot) ---------------

# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5

# Marker counts drive rotation directly in this variant.
MARKER_COUNTS = [4, 5, 6, 7, 8]
MARKER_COUNT_WEIGHTS = [2, 3, 3, 2, 1]
NORMAL_MAX = 6
COLORED_MAX = 2

# Color pool (all ARC colors except 1, which is reserved for T-marker cells)
COLOR_POOL = [c for c in range(10) if c != 1]

# Generation retry budget
SEED_ATTEMPTS = 200
GEN_RETRIES = 300

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _choose_layout_counts() -> tuple[int, int]:
    n_markers = random.choices(MARKER_COUNTS, weights=MARKER_COUNT_WEIGHTS)[0]
    region_choices = list(range(n_markers, 9))
    region_weights = list(range(len(region_choices), 0, -1))
    n_regions = random.choices(region_choices, weights=region_weights)[0]
    return n_regions, n_markers


def _choose_marker_counts(n_markers: int) -> tuple[int, int, int] | None:
    min_colored = max(0, int(n_markers) - NORMAL_MAX)
    max_colored = min(COLORED_MAX, int(n_markers) - 2)
    if min_colored > max_colored:
        return None
    n_colored = random.randint(min_colored, max_colored)
    n_normal = int(n_markers) - n_colored
    return int(n_markers), int(n_normal), int(n_colored)


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
                    a, b = int(region[r, c]), int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


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
            grid[center[0], center[1]] = tc
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[tuple], out_colors: list[int]) -> None:
    for _kind, center, stem, src, _dst, _tc in placed_markers:
        color = out_colors[src]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _gen_attempt() -> np.ndarray | None:
    n_regions, n_markers_target = _choose_layout_counts()
    counts = _choose_marker_counts(n_markers_target)
    if counts is None:
        return None
    n_markers, n_normal, n_colored = counts

    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    directed_pairs = _find_adjacent_pairs(region)
    random.shuffle(directed_pairs)
    dest_used = set()
    marker_pairs = []
    for src, dst in directed_pairs:
        if dst not in dest_used:
            marker_pairs.append((src, dst))
            dest_used.add(dst)
            if len(marker_pairs) >= n_markers:
                break
    if len(marker_pairs) != n_markers:
        return None
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
        _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, tc))

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
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
