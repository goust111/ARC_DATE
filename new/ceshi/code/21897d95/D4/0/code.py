# Variant of 21897d95
# Dimension: D4 — 标记方向判定机制 (src)
# Change: 茎方向为T中心的3个分支中唯一没有对称方向的那个；茎指向目标区域，茎方向+横杆方向的两个对角线单元在源区域中... → 茎长度编码优先级

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D4 variant
# ==================================
# a) Concretization and behavior trace
#    - A normal marker is still a T made of color 1, but its stem is now a straight
#      line of length 1..3 instead of a single cell. Shape size becomes 4..6 cells:
#      center + 2 bar cells + stem line.
#    - A colored-center marker keeps a non-1 center, two bar singletons of color 1,
#      and a color-1 stem line of length 1..3. The center color is still the literal
#      replacement color if that marker wins.
#    - Source identification stays geometric: use the two cells diagonally beside the
#      stem tip. Destination identification also stays geometric: walk forward from the
#      center along the stem and take the first non-control component after the tip.
#    - New behavior appears only when several markers point at the same destination.
#      Example: a length-1 normal T from red->blue and a length-3 normal T from green->blue.
#      The old code would either forbid the conflict or silently depend on iteration order.
#      The variant instead chooses the length-3 marker, so blue becomes green.
#
# b) Complexity calibration
#    - Stem length has only three values {1,2,3}, which is enough to make the change
#      visible while keeping the search space human-readable.
#    - The generator always includes at least one repeated-destination competition with
#      different stem lengths, so the task does not collapse back to the original logic.
#
# c) Old implicit assumptions that break under the variant
#    - `_parse_tetromino` assumed every normal marker has exactly 4 cells: invalid now.
#    - `first_component_from(..., start=2)` assumed the tip is always one step away:
#      now the first destination cell starts at `stem_len + 1`.
#    - `center + stem + bar` assumed the source diagonals are always beside a 1-cell tip:
#      now they must be computed from the actual tip at `center + stem_len * stem`.
#    - `final_color[dest] = replacement` for every marker assumed there is at most one
#      meaningful marker per destination: now only the longest stem may write that value.
#
# d) Placement legality
#    - Every marker reserves its own cells plus a 4-neighborhood buffer. This prevents
#      accidental merging of color-1 components, preserves colored-marker singleton bars,
#      keeps source-diagonal reference cells free, and leaves the solver with unambiguous
#      shapes.
#
# e) Visual observability
#    - Priority depends only on visible stem length, never on hidden color ids or scan
#      order. Competing markers are generated with different source/replacement colors so
#      the winning choice changes the output in a visible way.
#
# f) Visual cleanliness
#    - Markers never overlap and are buffered apart. Only one destination per competition
#      group is duplicated, so the grid remains readable instead of becoming cluttered.
#
# g) Color-change inferability
#    - Region colors remain distinct in the input. Recoloring still follows either
#      "destination takes source color" or "destination takes colored-center value"; the
#      only new variable is which marker wins, and that is signaled by stem length.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
STEM_LENGTHS = (1, 2, 3)


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


def _stem_cells(center: tuple[int, int], stem: tuple[int, int], stem_len: int) -> set[tuple[int, int]]:
    r, c = center
    return {
        (int(r + stem[0] * step), int(c + stem[1] * step))
        for step in range(1, int(stem_len) + 1)
    }


def _marker_cells(center: tuple[int, int], stem: tuple[int, int], stem_len: int) -> set[tuple[int, int]]:
    r, c = center
    cells = {(int(r), int(c))}
    cells |= _stem_cells(center, stem, stem_len)
    for dr, dc in _get_bar(stem):
        cells.add((int(r + dr), int(c + dc)))
    return cells


def _colored_one_cells(center: tuple[int, int], stem: tuple[int, int], stem_len: int) -> set[tuple[int, int]]:
    r, c = center
    cells = _stem_cells(center, stem, stem_len)
    for dr, dc in _get_bar(stem):
        cells.add((int(r + dr), int(c + dc)))
    return cells


def _t_neighborhood(center: tuple[int, int], stem: tuple[int, int], stem_len: int) -> set[tuple[int, int]]:
    cells = _marker_cells(center, stem, stem_len)
    neighborhood: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (int(r + dr), int(c + dc))
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _source_diagonal_cells(
    center: tuple[int, int], stem: tuple[int, int], stem_len: int
) -> list[tuple[int, int]]:
    tip = (
        int(center[0] + stem[0] * stem_len),
        int(center[1] + stem[1] * stem_len),
    )
    return [(int(tip[0] + dr), int(tip[1] + dc)) for dr, dc in _get_bar(stem)]


def _parse_normal_t_component(
    cells: set[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], int] | None:
    if len(cells) < 4 or len(cells) > 6:
        return None
    for r, c in cells:
        branches = [(dr, dc) for dr, dc in DIRS if (r + dr, c + dc) in cells]
        if len(branches) != 3:
            continue
        try:
            stem = _stem_from_branches(branches)
        except AssertionError:
            continue
        bar = [d for d in branches if d != stem]
        if len(bar) != 2 or bar[0] != _opp(bar[1]):
            continue
        stem_len = 0
        while (r + stem[0] * (stem_len + 1), c + stem[1] * (stem_len + 1)) in cells:
            stem_len += 1
        if stem_len not in STEM_LENGTHS:
            continue
        expected = _marker_cells((int(r), int(c)), stem, stem_len)
        if cells == expected:
            return (int(r), int(c)), stem, bar, stem_len
    return None


def _parse_stem_component(
    cells: set[tuple[int, int]], center: tuple[int, int], stem: tuple[int, int]
) -> int | None:
    stem_len = 0
    while (center[0] + stem[0] * (stem_len + 1), center[1] + stem[1] * (stem_len + 1)) in cells:
        stem_len += 1
    if stem_len not in STEM_LENGTHS:
        return None
    return stem_len if cells == _stem_cells(center, stem, stem_len) else None


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
        parsed = _parse_normal_t_component(cells)
        if parsed is None:
            continue
        center, stem, bar, stem_len = parsed
        markers.append(
            {
                "kind": "normal",
                "center": center,
                "stem": stem,
                "bar": bar,
                "stem_len": stem_len,
                "cells": set(cells),
            }
        )
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            branches = [(dr, dc) for dr, dc in DIRS if _in_bounds(g.shape, r + dr, c + dc) and int(g[r + dr, c + dc]) == 1]
            if len(branches) != 3:
                continue
            try:
                stem = _stem_from_branches(branches)
            except AssertionError:
                continue
            bar = [d for d in branches if d != stem]
            if len(bar) != 2 or bar[0] != _opp(bar[1]):
                continue

            stem_front = (int(r + stem[0]), int(c + stem[1]))
            stem_label = int(labels_1[stem_front])
            if stem_label <= 0 or one_components[stem_label] & used_ones:
                continue
            stem_len = _parse_stem_component(one_components[stem_label], (int(r), int(c)), stem)
            if stem_len is None:
                continue

            bar_cells: list[tuple[int, int]] = []
            seen_labels = {stem_label}
            ok = True
            for dr, dc in bar:
                rr, cc = int(r + dr), int(c + dc)
                label = int(labels_1[rr, cc])
                if label <= 0 or one_sizes[label] != 1 or (rr, cc) in used_ones or label in seen_labels:
                    ok = False
                    break
                seen_labels.add(label)
                bar_cells.append((rr, cc))
            if not ok:
                continue

            marker_cells = set(bar_cells) | one_components[stem_label] | {(int(r), int(c))}
            markers.append(
                {
                    "kind": "colored",
                    "center": (int(r), int(c)),
                    "stem": stem,
                    "bar": bar,
                    "stem_len": stem_len,
                    "cells": marker_cells,
                    "target_color": int(g[r, c]),
                }
            )
            used_ones |= one_components[stem_label]
            used_ones |= set(bar_cells)
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
            stem_len = int(marker["stem_len"])
            dest = first_component_from(center, stem, start=stem_len + 1)
            assert dest is not None, "T marker must point at a destination region"

            source_parts: list[int] = []
            for rr, cc in _source_diagonal_cells(center, stem, stem_len):
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
    winning_markers: dict[int, dict] = {}
    for marker in markers:
        source = dsu.find(marker["source_comp"])
        marker["source"] = source
        if marker["kind"] not in {"normal", "colored"}:
            continue
        dest = dsu.find(marker["dest_comp"])
        replacement = int(marker["target_color"]) if marker["kind"] == "colored" else int(root_color[source])
        marker["dest"] = dest
        marker["replacement"] = replacement
        current = winning_markers.get(dest)
        if current is None or int(marker["stem_len"]) > int(current["stem_len"]):
            winning_markers[dest] = marker
        elif int(marker["stem_len"]) == int(current["stem_len"]):
            assert int(current["replacement"]) == replacement, "conflicting equal-priority T controls for one region"

    for dest, marker in winning_markers.items():
        final_color[int(dest)] = int(marker["replacement"])

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
    Repaint mosaic regions from T controls whose stem length now encodes priority, then optionally rotate using a color-9 L marker.

    Input structure:
    - The grid is partitioned into monochrome regions of non-1 colors.
    - Color 1 still draws T-shaped controls on region boundaries, but the stem can be 1, 2, or 3 cells long.
    - Some controls use a colored center plus color-1 bar/stem cells; that center color is the literal repaint value if the marker wins.
    - An optional 5-cell color-9 L marker still signals a 90-degree rotation of the solved grid.

    Rule:
    - The T center is the junction with three immediate branches; the unmatched branch direction is the stem direction.
    - The destination is the first region found straight ahead beyond the full stem, while the two cells diagonally beside the stem tip identify the source side.
    - If several T markers point to the same destination, only the marker with the longest stem applies its recoloring; shorter competitors are ignored for that destination.
    - Every marker cell is erased by filling it with the final color of its source-side region, and the L marker rotates the finished grid as before.

    Key insight:
    - Direction still says which region pair a marker connects, but now visible stem length decides which competing instruction has priority.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (l9rot + long-stem priority) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

COLOR_POOL = [c for c in range(10) if c != 1]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 300


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


def _build_adjacency(region: np.ndarray) -> dict[int, list[int]]:
    adjacency: dict[int, set[int]] = {}
    H, W = region.shape
    for r in range(H):
        for c in range(W):
            a = int(region[r, c])
            adjacency.setdefault(a, set())
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    b = int(region[nr, nc])
                    if a != b:
                        adjacency[a].add(b)
    return {dst: sorted(srcs) for dst, srcs in adjacency.items()}


def _choose_marker_counts(n_regions: int) -> tuple[int, int, int] | None:
    max_possible = min(MARKER_MAX, n_regions + 1)
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
    if n_normal < 2:
        return None
    return n_markers, n_normal, n_colored


def _marker_replacement(marker_def: dict, input_colors: list[int]) -> int:
    if marker_def["kind"] == "colored":
        return int(marker_def["target_color"])
    return int(input_colors[marker_def["src"]])


def _winning_replacements(marker_defs: list[dict], input_colors: list[int]) -> dict[int, int]:
    winners: dict[int, dict] = {}
    for marker_def in marker_defs:
        dst = int(marker_def["dst"])
        replacement = _marker_replacement(marker_def, input_colors)
        current = winners.get(dst)
        if current is None or int(marker_def["stem_len"]) > int(current["stem_len"]):
            winners[dst] = {"stem_len": int(marker_def["stem_len"]), "replacement": replacement}
        elif int(marker_def["stem_len"]) == int(current["stem_len"]):
            assert int(current["replacement"]) == replacement
    return {dst: int(info["replacement"]) for dst, info in winners.items()}


def _build_marker_defs(region: np.ndarray, input_colors: list[int]) -> list[dict] | None:
    n_regions = len(input_colors)
    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    n_markers, n_normal, n_colored = counts

    adjacency = _build_adjacency(region)
    competition_dsts = [dst for dst, srcs in adjacency.items() if len(srcs) >= 2]
    if not competition_dsts:
        return None

    competition_dst = random.choice(competition_dsts)
    other_dsts = [dst for dst in range(n_regions) if dst != competition_dst and adjacency.get(dst)]
    if len(other_dsts) < n_markers - 2:
        return None
    chosen_other_dsts = random.sample(other_dsts, n_markers - 2)

    marker_defs: list[dict] = []
    winner_src, loser_src = random.sample(adjacency[competition_dst], 2)
    loser_len = random.randint(1, 2)
    winner_len = random.randint(loser_len + 1, 3)
    marker_defs.append(
        {"kind": "normal", "src": int(winner_src), "dst": int(competition_dst), "target_color": None, "stem_len": winner_len}
    )
    marker_defs.append(
        {"kind": "normal", "src": int(loser_src), "dst": int(competition_dst), "target_color": None, "stem_len": loser_len}
    )

    colored_dsts = set(random.sample(chosen_other_dsts, n_colored))
    for dst in chosen_other_dsts:
        src = random.choice(adjacency[dst])
        stem_len = random.choice(STEM_LENGTHS)
        if dst in colored_dsts:
            cands = [c for c in COLOR_POOL if c != input_colors[dst]]
            if not cands:
                return None
            marker_defs.append(
                {
                    "kind": "colored",
                    "src": int(src),
                    "dst": int(dst),
                    "target_color": int(random.choice(cands)),
                    "stem_len": int(stem_len),
                }
            )
        else:
            marker_defs.append(
                {"kind": "normal", "src": int(src), "dst": int(dst), "target_color": None, "stem_len": int(stem_len)}
            )

    assert len(marker_defs) == n_markers
    assert sum(1 for d in marker_defs if d["kind"] == "normal") == n_normal
    assert sum(1 for d in marker_defs if d["kind"] == "colored") == n_colored
    return marker_defs


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    stem_len: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            center = (int(r), int(c))
            for stem in DIRS:
                cells = _marker_cells(center, stem, stem_len)
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                neighborhood = _t_neighborhood(center, stem, stem_len)
                if cells & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue
                if neighborhood & occupied_cells:
                    continue
                diagonals = _source_diagonal_cells(center, stem, stem_len)
                if not all(
                    0 <= rr < H and 0 <= cc < W and int(region[rr, cc]) == int(src)
                    for rr, cc in diagonals
                ):
                    continue
                step = stem_len + 1
                found = False
                while True:
                    nr, nc = r + stem[0] * step, c + stem[1] * step
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied_cells or (nr, nc) in cells:
                        step += 1
                        continue
                    found = int(region[nr, nc]) == int(dst)
                    break
                if found:
                    candidates.append((center, stem))
    return candidates


def _register_marker(
    center: tuple[int, int],
    stem: tuple[int, int],
    stem_len: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
    cells = _marker_cells(center, stem, stem_len)
    occupied_cells |= cells
    occupied_neighbors |= (_t_neighborhood(center, stem, stem_len) - cells)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    return np.asarray(colors, dtype=int)[region]


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[dict]) -> None:
    for marker in placed_markers:
        center = marker["center"]
        stem = marker["stem"]
        stem_len = int(marker["stem_len"])
        if marker["kind"] == "normal":
            for cr, cc in _marker_cells(center, stem, stem_len):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = int(marker["target_color"])
            for cr, cc in _colored_one_cells(center, stem, stem_len):
                grid[cr, cc] = 1


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[dict], out_colors: list[int]) -> None:
    for marker in placed_markers:
        color = int(out_colors[marker["src"]])
        for cr, cc in _marker_cells(marker["center"], marker["stem"], int(marker["stem_len"])):
            grid[cr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[dict],
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
    for marker in placed_markers:
        all_marker_cells |= _marker_cells(marker["center"], marker["stem"], int(marker["stem_len"]))

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


def _extract_markers_for_validation(g: np.ndarray) -> list[dict]:
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
        parsed = _parse_normal_t_component(cells)
        if parsed is None:
            continue
        center, stem, bar, stem_len = parsed
        diagonals = _source_diagonal_cells(center, stem, stem_len)
        replacement = int(g[diagonals[0][0], diagonals[0][1]])
        markers.append(
            {
                "kind": "normal",
                "center": center,
                "stem": stem,
                "stem_len": stem_len,
                "cells": set(cells),
                "replacement": replacement,
            }
        )
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            branches = [(dr, dc) for dr, dc in DIRS if _in_bounds(g.shape, r + dr, c + dc) and int(g[r + dr, c + dc]) == 1]
            if len(branches) != 3:
                continue
            try:
                stem = _stem_from_branches(branches)
            except AssertionError:
                continue
            bar = [d for d in branches if d != stem]
            if len(bar) != 2 or bar[0] != _opp(bar[1]):
                continue
            stem_front = (int(r + stem[0]), int(c + stem[1]))
            stem_label = int(labels_1[stem_front])
            if stem_label <= 0 or one_components[stem_label] & used_ones:
                continue
            stem_len = _parse_stem_component(one_components[stem_label], (int(r), int(c)), stem)
            if stem_len is None:
                continue
            seen_labels = {stem_label}
            bar_cells: list[tuple[int, int]] = []
            ok = True
            for dr, dc in bar:
                rr, cc = int(r + dr), int(c + dc)
                label = int(labels_1[rr, cc])
                if label <= 0 or one_sizes[label] != 1 or (rr, cc) in used_ones or label in seen_labels:
                    ok = False
                    break
                seen_labels.add(label)
                bar_cells.append((rr, cc))
            if not ok:
                continue
            marker_cells = set(bar_cells) | one_components[stem_label] | {(int(r), int(c))}
            markers.append(
                {
                    "kind": "colored",
                    "center": (int(r), int(c)),
                    "stem": stem,
                    "stem_len": stem_len,
                    "cells": marker_cells,
                    "replacement": int(g[r, c]),
                }
            )
            used_ones |= one_components[stem_label]
            used_ones |= set(bar_cells)
            for rr, cc in marker_cells:
                control[rr, cc] = True

    comp_id, _, _ = _label_base_components(g, control)

    for marker in markers:
        dest = None
        step = int(marker["stem_len"]) + 1
        center = marker["center"]
        stem = marker["stem"]
        while True:
            rr = center[0] + stem[0] * step
            cc = center[1] + stem[1] * step
            if not _in_bounds(g.shape, rr, cc):
                break
            if comp_id[rr, cc] >= 0:
                dest = int(comp_id[rr, cc])
                break
            step += 1
        marker["dest"] = dest
    return markers


def _has_decisive_priority_case(g: np.ndarray) -> bool:
    markers = _extract_markers_for_validation(np.asarray(g, dtype=int))
    by_dest: dict[int | None, list[dict]] = {}
    for marker in markers:
        by_dest.setdefault(marker["dest"], []).append(marker)
    for ms in by_dest.values():
        if len(ms) < 2:
            continue
        lengths = {int(m["stem_len"]) for m in ms}
        replacements = {int(m["replacement"]) for m in ms}
        if len(lengths) >= 2 and len(replacements) >= 2:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    marker_defs = _build_marker_defs(region, input_colors)
    if marker_defs is None:
        return None

    winners = _winning_replacements(marker_defs, input_colors)
    out_colors = list(input_colors)
    for dst, replacement in winners.items():
        out_colors[int(dst)] = int(replacement)

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers: list[dict] = []
    ordered_defs = list(marker_defs)
    random.shuffle(ordered_defs)
    ordered_defs.sort(key=lambda d: int(d["stem_len"]), reverse=True)
    for marker_def in ordered_defs:
        cands = _find_valid_placements(
            region,
            int(marker_def["src"]),
            int(marker_def["dst"]),
            int(marker_def["stem_len"]),
            occupied_cells,
            occupied_neighbors,
            H,
            W,
        )
        if not cands:
            return None
        center, stem = random.choice(cands)
        placed = dict(marker_def)
        placed["center"] = center
        placed["stem"] = stem
        _register_marker(center, stem, int(marker_def["stem_len"]), occupied_cells, occupied_neighbors)
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
    if not np.array_equal(result, output_grid):
        return None
    return input_grid if _has_decisive_priority_case(input_grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
