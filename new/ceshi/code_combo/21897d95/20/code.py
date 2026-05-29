# Combo variant of 21897d95 (index 20)
# combo_spec: D12/2+D14/2
# Name: 全局色组传播

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import label_mask
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import label_mask


# Design analysis for combo D12/2 + D14/2
# ---------------------------------------
# a) Concrete behavior and trace
# - The original task treats the T stem's first touched 4-connected component as
#   the repaint destination. The D12/2 single change broadens that from one local
#   component to a same-color propagation set; D14/2 broadens the identity rule
#   further so disconnected same-color islands are still one semantic region.
# - The fused rule is therefore not "follow connectivity until it stops"; it is
#   "read one destination color, then globally replace every non-control cell of
#   that color anywhere in the board".
# - Concrete example: suppose teal appears as two disconnected islands A and B.
#   A normal T points at island A, while its source diagonals identify orange.
#   Original 21897d95: only A becomes orange. Fused combo: A and B both become
#   orange, and the T cells themselves are also filled with the source side's
#   final orange. The visible difference is that a single local T causes several
#   distant patches with the same input color to change together.
#
# b) Complexity calibration
# - If repeated colors were optional, many generated boards would collapse back
#   to the original puzzle. The generator therefore always creates at least one
#   repeated-color group with two disconnected fragments and guarantees that at
#   least one T marker targets that group.
# - To keep the board legible, only 1-2 groups are split, each split into two
#   medium fragments rather than many tiny scraps. The base color vocabulary is
#   limited to 4-6 semantic groups so repeated colors are visible by inspection.
#
# c) Implicit-assumption audit of the original main()
# - Original assumption: connected-component ids are the semantic regions.
#   Broken: under the combo, semantic ownership is keyed by raw color instead.
# - Original assumption: DSU merges are needed to reconnect marker-cut pieces of
#   one region. Broken: the whole color group is already defined globally, so the
#   correct semantic id is just the color value itself.
# - Original assumption: `first_component_from()` may stop at one component and
#   recolor only that component. Broken: the first touched component only serves
#   to read the destination color group; the recolor target is every component of
#   that color.
# - Original assumption: diagonal source checks compare component identity after
#   DSU unions. Broken: in the combo they only need to agree on source color.
# - Original assumption: destination uniqueness is enforced per component.
#   Adapted: this combo still keeps at most one T per destination color group in
#   the generator, because the intended novelty is global propagation, not tie
#   breaking between several controls.
#
# d) Placement legality
# - Disconnected fragments that share a color are forced to be non-adjacent at
#   the fragment graph level; otherwise they would merge into one component and
#   the combo would lose its visible effect.
# - T markers remain pairwise disjoint and buffered exactly as in the original
#   family so color-1 tetrominoes do not fuse, and colored-center branches stay
#   singleton color-1 components.
# - The optional L-shaped color-9 marker is placed only inside one fragment and
#   outside every T neighborhood. Base region colors exclude 9 to keep the L cue
#   visually distinct.
#
# e) Visual observability
# - The key cue is repeated input color: several disconnected fragments share one
#   color, and a T pointing at one such fragment makes all of them change in the
#   output. This is directly inferable from 2-4 examples without hidden numeric
#   ordering of color ids.
# - At least one mandatory target group has an untouched sister fragment far from
#   the T itself, so the global effect cannot be mistaken for local boundary fill.
#
# f) Visual cleanliness
# - The generator avoids using color 9 for ordinary regions or colored T centers,
#   so the rotation marker never blends into the base palette.
# - Split groups are limited to disconnected fragments of normal region size, not
#   peppered noise. T markers keep one-cell buffers, so the grid still reads as
#   "mosaic + controls" rather than overlapping annotations.
#
# g) Color-change inferability
# - Normal T markers remain the primary signal for the combo because they copy an
#   existing visible source color. That makes the global replacement easier to
#   infer than if the mandatory propagated marker used an arbitrary center color.
# - Colored-center Ts are still allowed, but only as secondary variation. Their
#   center colors are chosen from colors unused by the base groups and distinct
#   from the destination's old color.
#
# h) New solver assumptions introduced here
# - Every same-color repeated group must be intentionally disconnected; generator
#   construction enforces this by pairing only non-adjacent fragments.
# - At least one targeted destination group must have >1 fragment and must change
#   to a different color; generator enforces that with one mandatory split-group
#   destination and globally distinct input colors per semantic group.
# - The small-fragment/L9 absorption logic still assumes a <=5-cell non-rectangular
#   control blob touches exactly one owner color group; generator placement keeps
#   the L marker embedded inside one fragment so that assumption remains valid.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


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


def _collect_t_markers(g: np.ndarray) -> tuple[list[dict], np.ndarray]:
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
        markers.append(
            {"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells}
        )
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
    return markers, control


def _add_small_markers(g: np.ndarray, markers: list[dict], control: np.ndarray) -> None:
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


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
    markers, control = _collect_t_markers(g)
    _add_small_markers(g, markers, control)

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base color group is required"

    color_areas: Counter[int] = Counter()
    for area, col in zip(comp_areas, comp_colors):
        color_areas[int(col)] += int(area)

    def first_group_from(center: tuple[int, int], direction: tuple[int, int], start: int) -> int | None:
        r, c = center
        step = int(start)
        while True:
            rr = r + direction[0] * step
            cc = c + direction[1] * step
            if not _in_bounds(g.shape, rr, cc):
                return None
            cid = int(comp_id[rr, cc])
            if cid >= 0:
                return int(comp_colors[cid])
            step += 1

    def adjacent_group_counts(cells: set[tuple[int, int]]) -> Counter[int]:
        counts: Counter[int] = Counter()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0:
                    counts[int(comp_colors[int(comp_id[rr, cc])])] += 1
        return counts

    for marker in markers:
        if marker["kind"] in {"normal", "colored"}:
            center = marker["center"]
            stem = marker["stem"]
            dest_color = first_group_from(center, stem, start=2)
            assert dest_color is not None, "T marker must point at a destination color group"

            source_colors: list[int] = []
            for bar_dir in marker["bar"]:
                rr = center[0] + stem[0] + bar_dir[0]
                cc = center[1] + stem[1] + bar_dir[1]
                assert _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0
                source_colors.append(int(comp_colors[int(comp_id[rr, cc])]))
            source_color = source_colors[0]
            for other in source_colors[1:]:
                assert other == source_color, "T source diagonals must share one source color group"

            marker["dest_color"] = int(dest_color)
            marker["source_color"] = int(source_color)
        else:
            counts = adjacent_group_counts(marker["cells"])
            assert counts, "small marker must touch its owner color group"
            owner = max(counts, key=lambda col: (counts[col], color_areas[col]))
            marker["source_color"] = int(owner)

    final_color = {int(col): int(col) for col in set(comp_colors)}
    targeted: set[int] = set()
    for marker in markers:
        if marker["kind"] not in {"normal", "colored"}:
            continue
        dest = int(marker["dest_color"])
        source = int(marker["source_color"])
        replacement = int(marker["target_color"]) if marker["kind"] == "colored" else source
        assert dest not in targeted, "generator/legal inputs use at most one T per destination color group"
        targeted.add(dest)
        final_color[dest] = replacement

    out = np.full(g.shape, -1, dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[int(comp_colors[cid])])

    for marker in markers:
        color = int(final_color[int(marker["source_color"])])
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
    Repaint global color groups from T-shaped control markers, erase the controls, and optionally rotate based on a color-9 L-marker.

    Input structure:
    - The board is tiled by monochrome fragments, and equal colors may appear in several disconnected places; those equal-colored fragments belong to one semantic color group.
    - Color 1 draws normal T controls on fragment boundaries.
    - Some controls use three singleton color-1 branches around a colored center; that center color is a literal replacement color.
    - An optional 5-cell L marker of color 9 is embedded inside one fragment and signals a 90-degree rotation of the finished board.

    Rule:
    - For a normal or colored T, the stem points toward a destination color group: the solver looks forward to the first non-control base cell, reads its color, and applies the repaint to every non-control cell of that color across the whole grid.
    - The two diagonal cells behind the stem identify the source color group; a normal T copies that source group's original input color, while a colored-center T uses the center color instead.
    - After the simultaneous recoloring step, every T cell and every absorbed small control blob is filled with the source side's final color.
    - A color-9 L marker rotates the final output: ┐ means 90 degrees counterclockwise, and ┌ means 90 degrees clockwise.

    Key insight:
    - Connectivity only supplies local geometry for locating stems and diagonals; the actual repaint target is global color identity, so one T can recolor several disconnected islands at once.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (global_color_group_propagation) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.6
GROUP_COUNTS = [4, 5, 6]
GROUP_WEIGHTS = [3, 3, 2]
EXTRA_FRAGMENT_COUNTS = [1, 2]
EXTRA_FRAGMENT_WEIGHTS = [3, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 450


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


def _find_fragment_adjacency(region: np.ndarray) -> dict[int, set[int]]:
    n_regions = int(region.max()) + 1
    adjacency = {rid: set() for rid in range(n_regions)}
    H, W = region.shape
    for r in range(H):
        for c in range(W):
            a = int(region[r, c])
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    b = int(region[nr, nc])
                    if a != b:
                        adjacency[a].add(b)
    return adjacency


def _choose_split_pairs(
    n_regions: int, adjacency: dict[int, set[int]], n_pairs: int
) -> list[tuple[int, int]] | None:
    candidates = [
        (a, b)
        for a in range(n_regions)
        for b in range(a + 1, n_regions)
        if b not in adjacency[a]
    ]
    random.shuffle(candidates)

    def backtrack(idx: int, used: set[int], chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == n_pairs:
            return list(chosen)
        if idx >= len(candidates):
            return None
        needed = n_pairs - len(chosen)
        available = sum(1 for a, b in candidates[idx:] if a not in used and b not in used)
        if available < needed:
            return None
        for j in range(idx, len(candidates)):
            a, b = candidates[j]
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            chosen.append((a, b))
            found = backtrack(j + 1, used, chosen)
            if found is not None:
                return found
            chosen.pop()
            used.remove(a)
            used.remove(b)
        return None

    return backtrack(0, set(), [])


def _assign_groups(
    n_regions: int, split_pairs: list[tuple[int, int]]
) -> tuple[list[int], list[int]]:
    groups: list[list[int]] = []
    used = set()
    for a, b in split_pairs:
        groups.append([a, b])
        used.add(a)
        used.add(b)
    for rid in range(n_regions):
        if rid not in used:
            groups.append([rid])
    random.shuffle(groups)

    region_group = [-1] * n_regions
    split_groups: list[int] = []
    for gid, members in enumerate(groups):
        for rid in members:
            region_group[rid] = gid
        if len(members) > 1:
            split_groups.append(gid)
    assert all(gid >= 0 for gid in region_group)
    return region_group, split_groups


def _choose_marker_counts(n_groups: int) -> tuple[int, int, int] | None:
    max_possible = min(MARKER_MAX, n_groups)
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


def _find_group_adjacency(
    adjacency: dict[int, set[int]], region_group: list[int]
) -> dict[int, set[int]]:
    n_groups = max(region_group) + 1
    group_adj = {gid: set() for gid in range(n_groups)}
    for rid, neighbors in adjacency.items():
        g0 = int(region_group[rid])
        for nb in neighbors:
            g1 = int(region_group[nb])
            if g0 != g1:
                group_adj[g0].add(g1)
    return group_adj


def _choose_marker_defs(
    group_adj: dict[int, set[int]],
    split_groups: list[int],
    n_markers: int,
    n_colored: int,
) -> list[tuple[str, int, int, int | None]] | None:
    groups = list(group_adj)
    if not split_groups:
        return None
    mandatory_dest = random.choice(split_groups)
    others = [g for g in groups if g != mandatory_dest]
    if len(others) < n_markers - 1:
        return None
    dest_groups = [mandatory_dest] + random.sample(others, n_markers - 1)
    random.shuffle(dest_groups)

    colored_pool = [g for g in dest_groups if g != mandatory_dest]
    if len(colored_pool) < n_colored:
        return None
    colored_dest_groups = set(random.sample(colored_pool, n_colored))

    defs: list[tuple[str, int, int, int | None]] = []
    for dst_group in dest_groups:
        src_choices = list(group_adj[dst_group])
        if not src_choices:
            return None
        src_group = random.choice(src_choices)
        kind = "colored" if dst_group in colored_dest_groups else "normal"
        defs.append((kind, src_group, dst_group, None))
    random.shuffle(defs)
    return defs


def _find_valid_placements(
    group_grid: np.ndarray,
    src_group: int,
    dst_group: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in DIRS:
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
                    and int(group_grid[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == src_group
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
                    found = int(group_grid[nr, nc]) == dst_group
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
            for dr, dc in DIRS:
                nb = (br + dr, bc + dc)
                if nb != center and nb not in branch:
                    occupied_neighbors.add(nb)
        for dr, dc in DIRS:
            nb = (center[0] + dr, center[1] + dc)
            if nb not in branch:
                occupied_neighbors.add(nb)


def _build_color_grid(group_grid: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[group_grid]


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
    for _kind, center, stem, src_group, _dst_group, _tc in placed_markers:
        color = int(out_colors[src_group])
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    region_group: list[int],
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
    owner_group = int(region_group[rid])
    for dr, dc in offsets:
        input_grid[lr + dr, lc + dc] = 9
        output_grid[lr + dr, lc + dc] = int(out_colors[owner_group])
    return orient


def _global_effect_metric(
    region: np.ndarray,
    region_group: list[int],
    placed_markers: list[tuple],
) -> tuple[int, int]:
    all_marker_cells = set()
    for _kind, center, stem, *_rest in placed_markers:
        all_marker_cells |= _t_cells(center, stem)

    members: dict[int, list[int]] = {}
    for rid, gid in enumerate(region_group):
        members.setdefault(int(gid), []).append(int(rid))

    region_sizes = {rid: int(np.count_nonzero(region == rid)) for rid in range(int(region.max()) + 1)}
    max_extra = 0
    active_targets = 0

    for _kind, center, stem, _src_group, dst_group, _tc in placed_markers:
        step = 2
        touched_rid = None
        while True:
            rr = center[0] + stem[0] * step
            cc = center[1] + stem[1] * step
            if not _in_bounds(region.shape, rr, cc):
                break
            if (rr, cc) in all_marker_cells:
                step += 1
                continue
            touched_rid = int(region[rr, cc])
            break
        if touched_rid is None or int(region_group[touched_rid]) != int(dst_group):
            continue
        extra = sum(region_sizes[rid] for rid in members[int(dst_group)] if rid != touched_rid)
        if extra > 0:
            active_targets += 1
            max_extra = max(max_extra, int(extra))
    return int(max_extra), int(active_targets)


def _gen_attempt() -> np.ndarray | None:
    n_groups = random.choices(GROUP_COUNTS, weights=GROUP_WEIGHTS)[0]
    extra = random.choices(EXTRA_FRAGMENT_COUNTS, weights=EXTRA_FRAGMENT_WEIGHTS)[0]
    n_regions = n_groups + extra
    min_size = max(10, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    adjacency = _find_fragment_adjacency(region)
    split_pairs = _choose_split_pairs(n_regions, adjacency, extra)
    if split_pairs is None:
        return None
    region_group, split_groups = _assign_groups(n_regions, split_pairs)
    group_adj = _find_group_adjacency(adjacency, region_group)

    counts = _choose_marker_counts(n_groups)
    if counts is None:
        return None
    n_markers, _n_normal, n_colored = counts

    marker_defs = _choose_marker_defs(group_adj, split_groups, n_markers, n_colored)
    if marker_defs is None:
        return None

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_groups > len(pool):
        return None
    input_colors = pool[:n_groups]
    unused_colors = [c for c in pool if c not in input_colors]

    out_colors = list(input_colors)
    used_targets: set[int] = set()
    filled_defs: list[tuple[str, int, int, int | None]] = []
    for kind, src_group, dst_group, _tc in marker_defs:
        if kind == "normal":
            out_colors[dst_group] = int(input_colors[src_group])
            filled_defs.append((kind, src_group, dst_group, None))
            continue
        cands = [c for c in unused_colors if c != input_colors[dst_group] and c not in used_targets]
        if not cands:
            return None
        tc = random.choice(cands)
        used_targets.add(int(tc))
        out_colors[dst_group] = int(tc)
        filled_defs.append((kind, src_group, dst_group, int(tc)))

    region_group_grid = np.array(region_group, dtype=int)[region]
    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers = []
    random.shuffle(filled_defs)
    for kind, src_group, dst_group, tc in filled_defs:
        cands = _find_valid_placements(
            region_group_grid, src_group, dst_group, occupied_cells, occupied_neighbors, H, W
        )
        if not cands:
            return None
        center, stem = random.choice(cands)
        _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src_group, dst_group, tc))

    max_extra, active_targets = _global_effect_metric(region, region_group, placed_markers)
    if active_targets < 1 or max_extra < MIN_REGION_CELLS:
        return None

    input_grid = _build_color_grid(region_group_grid, input_colors)
    output_grid = _build_color_grid(region_group_grid, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    l9_orient = _try_place_l9(
        region, region_group, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors
    )
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
