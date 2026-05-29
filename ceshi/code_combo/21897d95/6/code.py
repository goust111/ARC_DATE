# Combo variant of 21897d95 (index 6)
# combo_spec: D1/3+D12/1
# Name: 投票竞争重着色
#
# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
# - The base task reads each T locally: the stem ray finds the destination
#   region, the two source-side diagonals identify the source region, and a
#   colored center overrides the source color.
# - This combo changes the moment when destination color is chosen. Multiple Ts
#   may now hit the same destination, and the destination no longer follows the
#   nearest or last write by default. Instead every T contributes one vote for
#   its replacement color; the destination takes the majority color. Distance is
#   only used inside a tied vote.
# - Small trace:
#   - Marker A is a near normal T at distance 2, so it votes for source color 4.
#   - Marker B and C are farther colored-center Ts at distance 4, and both vote
#   for center color 7.
#   - Original task: such a grid is illegal because one destination would be
#     written by conflicting Ts.
#   - D12-only variant: A would win because it is nearest.
#   - This combo: destination becomes color 7 because votes are {7:2, 4:1}.
#   - Visible difference: the closest stem can lose when two farther Ts agree.
# b) Complexity calibration
# - If shared-target groups were left random, many samples would collapse back to
#   the old behavior because every voter would accidentally demand the same
#   replacement, or because the nearest marker would already match the majority.
# - The generator therefore forces two readable collisions:
#   one three-way contest with a 2-vs-1 majority, plus one two-way contest whose
#   votes tie and must be broken by distance. This keeps both combo branches
#   visible instead of leaving nearest-only logic latent.
# - Any remaining markers stay unique-destination normal Ts so the puzzle keeps
#   the original visual vocabulary instead of turning every region into conflict.
# c) Implicit assumptions audit
# - `first_component_from()` can no longer stop at a component id; the distance
#   to the first base component is part of the semantics, because tied vote
#   colors need a geometric tiebreak.
# - The original `final_color[dest] in {root_color[dest], replacement}` assumes
#   at most one effective writer per destination. This is false now, so solving
#   must become two-phase: gather all contenders, then resolve by vote count and
#   distance.
# - The original generator assumed unique destinations when choosing marker
#   pairs. The combo deliberately breaks that assumption for one controlled
#   destination while keeping the rest unique for readability.
# - New solver assumption: if vote leaders tie, the nearest marker among those
#   tied colors must identify a unique winning color. The generator enforces a
#   strict distance gap in its two-way tied contest, so no ambiguous
#   equal-distance cross-color tie is produced.
# d) Placement legality
# - T shapes must remain disjoint and non-touching so color-1 components do not
#   merge and colored-T branches stay singleton components.
# - Farther voters use tiny non-rectangular scrap blockers on the stem ray. Each
#   blocker is registered as control occupancy together with its T, so later
#   placements cannot overlap it or touch it.
# - The optional L9 rotation marker is also placed with the same occupancy
#   buffer, so it never collides with Ts or blocker scraps.
# e) Visual observability
# - The decisive variables are all visible: how many Ts point at one region,
#   which replacement color each T votes for, and how far each stem must travel.
# - No hidden ordering by numeric color id is used. Distance only compares
#   literal stem steps, and voting only compares repeated visible colors.
# f) Visual cleanliness
# - One destination is forced into a three-way contest and one into a two-way
#   tied contest. That is enough branch coverage to expose the rule while still
#   keeping the grid legible and region-based.
# - All colored voters share the same visible center color, and exactly two of
#   them point at the same destination. That gives the human solver a repeated
#   cue for "two votes for this color" instead of requiring them to infer a
#   latent category.
# g) Color-change inference
# - Region colors remain distinct, so a normal T still reads as "copy this
#   source region color". The new majority cue comes from repeated center colors
#   on the two colored Ts, not from arbitrary recolor noise.
# - Because the forced contest is 2-vs-1 and the lone near normal voter uses a
#   different replacement color, the majority result differs from the nearest-T
#   result on every generated grid. This is the metric used later to detect
#   degeneration.

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
        markers.append(
            {
                "kind": "normal",
                "center": center,
                "stem": stem,
                "bar": bar,
                "cells": cells,
            }
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

    def first_component_from(
        center: tuple[int, int], direction: tuple[int, int], start: int
    ) -> tuple[int | None, int | None]:
        r, c = center
        step = int(start)
        while True:
            rr = r + direction[0] * step
            cc = c + direction[1] * step
            if not _in_bounds(g.shape, rr, cc):
                return None, None
            if comp_id[rr, cc] >= 0:
                return int(comp_id[rr, cc]), int(step)
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
            dest, distance = first_component_from(center, stem, start=2)
            assert dest is not None and distance is not None, "T marker must point at a destination region"

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
            marker["dest_distance"] = distance
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
    contests: dict[int, list[dict]] = {}
    for marker in markers:
        marker["source"] = dsu.find(marker["source_comp"])
        if marker["kind"] not in {"normal", "colored"}:
            continue
        dest = dsu.find(marker["dest_comp"])
        marker["dest"] = dest
        marker["replacement"] = (
            int(marker["target_color"]) if marker["kind"] == "colored" else int(root_color[marker["source"]])
        )
        contests.setdefault(dest, []).append(marker)

    for dest, group in contests.items():
        votes = Counter(int(marker["replacement"]) for marker in group)
        best_votes = max(votes.values())
        top_colors = {color for color, count in votes.items() if int(count) == int(best_votes)}
        if len(top_colors) == 1:
            final_color[dest] = int(next(iter(top_colors)))
            continue
        tied_markers = [marker for marker in group if int(marker["replacement"]) in top_colors]
        best_distance = min(int(marker["dest_distance"]) for marker in tied_markers)
        nearest_colors = {
            int(marker["replacement"])
            for marker in tied_markers
            if int(marker["dest_distance"]) == int(best_distance)
        }
        assert len(nearest_colors) == 1, "tied vote leaders must have a unique nearest color"
        final_color[dest] = int(next(iter(nearest_colors)))

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
    Repaint regions from competing T-shaped controls, erase those controls, and optionally rotate after the repaint.

    Input structure:
    - The canvas is partitioned into connected monochrome regions with control shapes overlaid on top.
    - Normal controls are color-1 T tetrominoes; colored controls are T shapes with three singleton color-1 branches around a colored center.
    - Several Ts may point at the same destination region, and tiny non-rectangular scraps may sit on a stem ray as blocker controls.
    - An optional 5-cell color-9 L marker requests a 90-degree rotation of the finished output.

    Rule:
    - For every T, the unique stem direction points toward the destination region, while the two diagonals behind the stem identify the source-side region.
    - A normal T votes to recolor the destination to its source region color; a colored-center T votes to recolor the destination to the center color.
    - If multiple Ts target the same destination, count votes by replacement color and use the majority color. Only when the top vote count ties do you compare stem distances and keep the tied color whose marker reaches the destination first.
    - After all destination colors are resolved simultaneously, every control cell is filled with the final color of its owner region, and the color-9 L marker rotates the completed output: ┐ means CCW 90 degrees, ┌ means CW 90 degrees.

    Key insight:
    - A shared target is no longer resolved by a single local winner. You must read each T normally, then fuse all Ts aimed at the same region by visible color voting, using visible distance only as a tie-breaker.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (vote_competition) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 3.0
REGION_COUNTS = [4, 5, 6]
REGION_WEIGHTS = [4, 3, 2]

MARKER_MIN = 5
MARKER_MAX = 6
CONTEST_COLORED_COUNT = 2

REGION_COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 400


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


def _shape_neighborhood(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
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


def _shape_fits(
    shape: set[tuple[int, int]],
    occupied_controls: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
    H: int,
    W: int,
) -> bool:
    if not all(0 <= r < H and 0 <= c < W for r, c in shape):
        return False
    if shape & occupied_controls:
        return False
    if shape & occupied_buffer:
        return False
    if _shape_neighborhood(shape) & occupied_controls:
        return False
    return True


def _register_control_shape(
    shape: set[tuple[int, int]],
    occupied_controls: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
) -> None:
    occupied_controls |= shape
    occupied_buffer |= (_shape_neighborhood(shape) - shape)


def _find_valid_direct_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_controls: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[dict]:
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in DIRS:
                bar = _get_bar(stem)
                shape = _t_cells((r, c), stem)
                if not _shape_fits(shape, occupied_controls, occupied_buffer, H, W):
                    continue
                if not all(
                    0 <= r + stem[0] + bd[0] < H
                    and 0 <= c + stem[1] + bd[1] < W
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == int(src)
                    for bd in bar
                ):
                    continue
                step2 = (r + 2 * stem[0], c + 2 * stem[1])
                if not _in_bounds(region.shape, step2[0], step2[1]):
                    continue
                if step2 in occupied_controls:
                    continue
                if int(region[step2]) != int(dst):
                    continue
                candidates.append(
                    {
                        "src": int(src),
                        "dst": int(dst),
                        "center": (int(r), int(c)),
                        "stem": stem,
                        "distance": 2,
                        "shape": shape,
                        "scrap_cells": set(),
                    }
                )
    return candidates


def _find_valid_blocked_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_controls: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[dict]:
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in DIRS:
                bar = _get_bar(stem)
                t_shape = _t_cells((r, c), stem)
                if not _shape_fits(t_shape, occupied_controls, occupied_buffer, H, W):
                    continue
                if not all(
                    0 <= r + stem[0] + bd[0] < H
                    and 0 <= c + stem[1] + bd[1] < W
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == int(src)
                    for bd in bar
                ):
                    continue
                step2 = (r + 2 * stem[0], c + 2 * stem[1])
                step3 = (r + 3 * stem[0], c + 3 * stem[1])
                step4 = (r + 4 * stem[0], c + 4 * stem[1])
                if not all(_in_bounds(region.shape, rr, cc) for rr, cc in [step2, step3, step4]):
                    continue
                if not all(int(region[rr, cc]) == int(dst) for rr, cc in [step2, step3, step4]):
                    continue
                if step4 in occupied_controls:
                    continue
                for side in bar:
                    side_cell = (step2[0] + side[0], step2[1] + side[1])
                    if not _in_bounds(region.shape, side_cell[0], side_cell[1]):
                        continue
                    if int(region[side_cell]) != int(dst):
                        continue
                    scrap_cells = {step2, step3, side_cell}
                    shape = t_shape | scrap_cells
                    if not _shape_fits(shape, occupied_controls, occupied_buffer, H, W):
                        continue
                    candidates.append(
                        {
                            "src": int(src),
                            "dst": int(dst),
                            "center": (int(r), int(c)),
                            "stem": stem,
                            "distance": 4,
                            "shape": shape,
                            "scrap_cells": scrap_cells,
                        }
                    )
    return candidates


def _candidates_compatible(a: dict, b: dict) -> bool:
    shape_a = set(a["shape"])
    shape_b = set(b["shape"])
    if shape_a & shape_b:
        return False
    if shape_a & _shape_neighborhood(shape_b):
        return False
    if shape_b & _shape_neighborhood(shape_a):
        return False
    return True


def _collect_candidates_by_dst(
    region: np.ndarray, H: int, W: int
) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    direct_by_dst: dict[int, list[dict]] = {}
    blocked_by_dst: dict[int, list[dict]] = {}
    for src, dst in _find_adjacent_pairs(region):
        direct = _find_valid_direct_placements(region, src, dst, set(), set(), H, W)
        blocked = _find_valid_blocked_placements(region, src, dst, set(), set(), H, W)
        if direct:
            direct_by_dst.setdefault(int(dst), []).extend(direct)
        if blocked:
            blocked_by_dst.setdefault(int(dst), []).extend(blocked)
    return direct_by_dst, blocked_by_dst


def _select_majority_contest(
    direct_by_dst: dict[int, list[dict]], blocked_by_dst: dict[int, list[dict]]
) -> tuple[dict, dict, dict] | None:
    shared_dsts = [dst for dst in direct_by_dst if len(blocked_by_dst.get(dst, [])) >= CONTEST_COLORED_COUNT]
    random.shuffle(shared_dsts)
    for dst in shared_dsts:
        near_candidates = direct_by_dst[dst][:]
        far_candidates = blocked_by_dst[dst][:]
        random.shuffle(near_candidates)
        random.shuffle(far_candidates)
        for near in near_candidates:
            for i in range(len(far_candidates)):
                far_a = far_candidates[i]
                if not _candidates_compatible(near, far_a):
                    continue
                for j in range(i + 1, len(far_candidates)):
                    far_b = far_candidates[j]
                    if not _candidates_compatible(near, far_b):
                        continue
                    if not _candidates_compatible(far_a, far_b):
                        continue
                    return near, far_a, far_b
    return None


def _select_tie_contest(
    direct_by_dst: dict[int, list[dict]],
    blocked_by_dst: dict[int, list[dict]],
    reserved_dsts: set[int],
    incompatible_with: list[dict],
) -> tuple[dict, dict] | None:
    shared_dsts = [dst for dst in direct_by_dst if dst not in reserved_dsts and blocked_by_dst.get(dst)]
    random.shuffle(shared_dsts)
    for dst in shared_dsts:
        near_candidates = direct_by_dst[dst][:]
        far_candidates = blocked_by_dst[dst][:]
        random.shuffle(near_candidates)
        random.shuffle(far_candidates)
        for near in near_candidates:
            if any(not _candidates_compatible(near, other) for other in incompatible_with):
                continue
            for far in far_candidates:
                if not _candidates_compatible(near, far):
                    continue
                if any(not _candidates_compatible(far, other) for other in incompatible_with):
                    continue
                return near, far
    return None


def _choose_marker_count() -> int:
    return random.randint(MARKER_MIN, MARKER_MAX)


def _choose_remaining_pairs(
    directed_pairs: list[tuple[int, int]],
    reserved_dsts: set[int],
    count: int,
) -> list[tuple[int, int]]:
    pairs = directed_pairs[:]
    random.shuffle(pairs)
    chosen: list[tuple[int, int]] = []
    used_dsts: set[int] = set()
    for src, dst in pairs:
        if int(dst) in reserved_dsts:
            continue
        if int(dst) in used_dsts:
            continue
        chosen.append((int(src), int(dst)))
        used_dsts.add(int(dst))
        if len(chosen) >= count:
            break
    return chosen


def _build_specs(
    region: np.ndarray,
    majority_contest: tuple[dict, dict, dict],
    tie_contest: tuple[dict, dict],
    n_markers: int,
    vote_color: int,
) -> list[dict] | None:
    near, far_a, far_b = majority_contest
    tie_near, tie_far = tie_contest
    directed_pairs = _find_adjacent_pairs(region)
    reserved_dsts = {int(near["dst"]), int(tie_near["dst"])}
    remaining = _choose_remaining_pairs(directed_pairs, reserved_dsts, n_markers - 5)
    if len(remaining) != n_markers - 5:
        return None
    specs = [
        {
            "kind": "normal",
            "src": int(near["src"]),
            "dst": int(near["dst"]),
            "forced": near,
            "target_color": None,
        },
        {
            "kind": "colored",
            "src": int(far_a["src"]),
            "dst": int(far_a["dst"]),
            "forced": far_a,
            "target_color": int(vote_color),
        },
        {
            "kind": "colored",
            "src": int(far_b["src"]),
            "dst": int(far_b["dst"]),
            "forced": far_b,
            "target_color": int(vote_color),
        },
        {
            "kind": "normal",
            "src": int(tie_near["src"]),
            "dst": int(tie_near["dst"]),
            "forced": tie_near,
            "target_color": None,
        },
        {
            "kind": "colored",
            "src": int(tie_far["src"]),
            "dst": int(tie_far["dst"]),
            "forced": tie_far,
            "target_color": int(vote_color),
        },
    ]
    for src, dst in remaining:
        specs.append(
            {
                "kind": "normal",
                "src": int(src),
                "dst": int(dst),
                "forced": None,
                "target_color": None,
            }
        )
    return specs


def _place_specs(
    region: np.ndarray, specs: list[dict], H: int, W: int
) -> tuple[list[dict], list[dict]] | None:
    occupied_controls: set[tuple[int, int]] = set()
    occupied_buffer: set[tuple[int, int]] = set()
    placed_markers: list[dict] = []
    placed_scraps: list[dict] = []

    forced_specs = [spec for spec in specs if spec["forced"] is not None]
    flexible_specs = [spec for spec in specs if spec["forced"] is None]
    random.shuffle(flexible_specs)

    for spec in forced_specs + flexible_specs:
        forced = spec["forced"]
        if forced is not None:
            shape = set(forced["shape"])
            if not _shape_fits(shape, occupied_controls, occupied_buffer, H, W):
                return None
            candidate = forced
        else:
            candidates = _find_valid_direct_placements(
                region,
                int(spec["src"]),
                int(spec["dst"]),
                occupied_controls,
                occupied_buffer,
                H,
                W,
            )
            if not candidates:
                return None
            candidate = random.choice(candidates)
            shape = set(candidate["shape"])

        _register_control_shape(shape, occupied_controls, occupied_buffer)
        placed_markers.append(
            {
                "kind": spec["kind"],
                "center": tuple(candidate["center"]),
                "stem": tuple(candidate["stem"]),
                "src": int(spec["src"]),
                "dst": int(spec["dst"]),
                "target_color": spec["target_color"],
                "distance": int(candidate["distance"]),
            }
        )
        if candidate["scrap_cells"]:
            placed_scraps.append(
                {
                    "cells": set(candidate["scrap_cells"]),
                    "owner": int(spec["dst"]),
                }
            )
    return placed_markers, placed_scraps


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    return np.asarray(colors, dtype=int)[region]


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[dict]) -> None:
    for marker in placed_markers:
        if marker["kind"] == "normal":
            for cr, cc in _t_cells(marker["center"], marker["stem"]):
                grid[cr, cc] = 1
        else:
            grid[marker["center"][0], marker["center"][1]] = int(marker["target_color"])
            for cr, cc in _t_branch_cells(marker["center"], marker["stem"]):
                grid[cr, cc] = 1


def _stamp_markers_output(grid: np.ndarray, placed_markers: list[dict], out_colors: list[int]) -> None:
    for marker in placed_markers:
        color = int(out_colors[int(marker["src"])])
        for cr, cc in _t_cells(marker["center"], marker["stem"]):
            grid[cr, cc] = color


def _stamp_scraps_input(grid: np.ndarray, scraps: list[dict], scrap_color: int) -> None:
    for scrap in scraps:
        for r, c in scrap["cells"]:
            grid[r, c] = int(scrap_color)


def _stamp_scraps_output(grid: np.ndarray, scraps: list[dict], out_colors: list[int]) -> None:
    for scrap in scraps:
        color = int(out_colors[int(scrap["owner"])])
        for r, c in scrap["cells"]:
            grid[r, c] = color


def _winner_map(placed_markers: list[dict], input_colors: list[int]) -> dict[int, int]:
    grouped: dict[int, list[dict]] = {}
    for marker in placed_markers:
        grouped.setdefault(int(marker["dst"]), []).append(marker)

    winners: dict[int, int] = {}
    for dst, group in grouped.items():
        votes = Counter(
            int(marker["target_color"]) if marker["kind"] == "colored" else int(input_colors[int(marker["src"])])
            for marker in group
        )
        best_votes = max(votes.values())
        top_colors = {color for color, count in votes.items() if int(count) == int(best_votes)}
        if len(top_colors) == 1:
            winners[int(dst)] = int(next(iter(top_colors)))
            continue
        best_distance = min(
            int(marker["distance"])
            for marker in group
            if (
                int(marker["target_color"]) if marker["kind"] == "colored" else int(input_colors[int(marker["src"])])
            ) in top_colors
        )
        nearest_colors = {
            int(marker["target_color"]) if marker["kind"] == "colored" else int(input_colors[int(marker["src"])])
            for marker in group
            if (
                (int(marker["target_color"]) if marker["kind"] == "colored" else int(input_colors[int(marker["src"])]))
                in top_colors
                and int(marker["distance"]) == int(best_distance)
            )
        }
        if len(nearest_colors) != 1:
            raise AssertionError("generator created an ambiguous tied vote")
        winners[int(dst)] = int(next(iter(nearest_colors)))
    return winners


def _control_occupancy_from_placements(
    placed_markers: list[dict], placed_scraps: list[dict]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    occupied_controls: set[tuple[int, int]] = set()
    occupied_buffer: set[tuple[int, int]] = set()
    for marker in placed_markers:
        _register_control_shape(_t_cells(marker["center"], marker["stem"]), occupied_controls, occupied_buffer)
    for scrap in placed_scraps:
        _register_control_shape(set(scrap["cells"]), occupied_controls, occupied_buffer)
    return occupied_controls, occupied_buffer


def _try_place_l9(
    region: np.ndarray,
    occupied_controls: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
) -> str | None:
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape
    candidates = []
    for r in range(H):
        for c in range(W):
            cells = {(r + dr, c + dc) for dr, dc in offsets}
            if not _shape_fits(cells, occupied_controls, occupied_buffer, H, W):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == int(rid) for cr, cc in cells):
                continue
            candidates.append((cells, rid))
    if not candidates:
        return None
    cells, rid = random.choice(candidates)
    for cr, cc in cells:
        input_grid[cr, cc] = 9
        output_grid[cr, cc] = int(out_colors[int(rid)])
    return orient


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(11, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    direct_by_dst, blocked_by_dst = _collect_candidates_by_dst(region, H, W)
    majority_contest = _select_majority_contest(direct_by_dst, blocked_by_dst)
    if majority_contest is None:
        return None
    near, far_a, _far_b = majority_contest
    tie_contest = _select_tie_contest(
        direct_by_dst,
        blocked_by_dst,
        {int(near["dst"])},
        [near, far_a, _far_b],
    )
    if tie_contest is None:
        return None
    tie_near, tie_far = tie_contest

    n_markers = _choose_marker_count()

    pool = list(REGION_COLOR_POOL)
    random.shuffle(pool)
    input_colors = pool[:n_regions]
    spare_colors = [c for c in REGION_COLOR_POOL if c not in input_colors]
    if len(spare_colors) < 2:
        return None

    near_color = int(input_colors[int(near["src"])])
    vote_candidates = [c for c in spare_colors if int(c) != int(near_color)]
    if not vote_candidates:
        return None
    vote_color = int(random.choice(vote_candidates))
    scrap_candidates = [c for c in spare_colors if int(c) != int(vote_color)]
    if not scrap_candidates:
        return None
    scrap_color = int(random.choice(scrap_candidates))

    specs = _build_specs(region, majority_contest, tie_contest, n_markers, vote_color)
    if specs is None:
        return None

    placed = _place_specs(region, specs, H, W)
    if placed is None:
        return None
    placed_markers, placed_scraps = placed

    shared_group = [m for m in placed_markers if int(m["dst"]) == int(near["dst"])]
    if len(shared_group) != 3:
        return None
    shared_votes = Counter(
        int(m["target_color"]) if m["kind"] == "colored" else int(input_colors[int(m["src"])])
        for m in shared_group
    )
    if sorted(shared_votes.values()) != [1, 2]:
        return None
    nearest_color = min(
        shared_group,
        key=lambda m: (int(m["distance"]), 0 if m["kind"] == "normal" else 1),
    )
    nearest_replacement = (
        int(nearest_color["target_color"])
        if nearest_color["kind"] == "colored"
        else int(input_colors[int(nearest_color["src"])])
    )
    majority_replacement = int(shared_votes.most_common(1)[0][0])
    if int(nearest_replacement) == int(majority_replacement):
        return None

    tie_group = [m for m in placed_markers if int(m["dst"]) == int(tie_near["dst"])]
    if len(tie_group) != 2:
        return None
    tie_votes = Counter(
        int(m["target_color"]) if m["kind"] == "colored" else int(input_colors[int(m["src"])])
        for m in tie_group
    )
    if sorted(tie_votes.values()) != [1, 1]:
        return None
    if {m["kind"] for m in tie_group} != {"normal", "colored"}:
        return None
    tie_winner = min(tie_group, key=lambda m: (int(m["distance"]), 0 if m["kind"] == "normal" else 1))
    tie_winner_replacement = (
        int(tie_winner["target_color"])
        if tie_winner["kind"] == "colored"
        else int(input_colors[int(tie_winner["src"])])
    )
    if int(tie_winner_replacement) == int(vote_color):
        return None

    winners = _winner_map(placed_markers, input_colors)
    out_colors = list(input_colors)
    for dst, replacement in winners.items():
        out_colors[int(dst)] = int(replacement)

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_scraps_input(input_grid, placed_scraps, scrap_color)
    _stamp_scraps_output(output_grid, placed_scraps, out_colors)
    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    occupied_controls, occupied_buffer = _control_occupancy_from_placements(placed_markers, placed_scraps)
    l9_orient = _try_place_l9(region, occupied_controls, occupied_buffer, input_grid, output_grid, out_colors)
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
