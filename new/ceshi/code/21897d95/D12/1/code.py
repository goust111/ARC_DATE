# Variant of 21897d95
# Dimension: D12 — T标记数量与区域的关系 (meta)
# Change: 4-8个T标记（2-6个normal + 0-2个colored），每个目标区域最多被一个T标记指向... → 允许多个T标记指向同一目标区域

# Design analysis for the D12 variant
# ----------------------------------
# a) Concrete behavior and trace
# - The original puzzle already lets a T marker define a destination by looking
#   forward along the stem and a source by reading the two diagonals behind the
#   stem. The new behavior keeps the same local geometry, but now two or more T
#   markers may all point at the same destination region.
# - To make the change visible, each generated grid contains at least one shared
#   target with two normal T markers from different source regions. One marker
#   reaches the target immediately at stem distance 2. The other has a tiny
#   non-rectangular scrap inserted in front of it, so the solver skips control
#   cells and reaches the same destination at distance 4.
# - Small trace:
#   - Old logic: marker A points to region X and marker B is forbidden from also
#     pointing to region X unless both demand the same color; otherwise the
#     solver asserts on "conflicting T controls for one region".
#   - New logic: marker A and marker B may both point to region X. The solver
#     records A.distance=2 and B.distance=4, groups them by destination, and
#     only A decides region X's new color because it is closer.
#   - Visible difference: both Ts visually aim at the same region, but only the
#     nearer source color appears inside that target region in the output.
# b) Complexity calibration
# - The variant would be invisible if repeated markers always had the same
#   source color or tied at the same distance. The generator therefore forces at
#   least one shared target with different source colors and different distances.
# - The variant would become noisy if many regions were hit by many unrelated
#   conflicts. The generator keeps one mandatory many:1 contest and leaves the
#   remaining markers close to the original style.
# c) Implicit assumptions in the original main()
# - The original `first_component_from()` only returned a destination id; D12
#   needs the number of stem steps as well.
# - The original `assert final_color[dest] in {root_color[dest], replacement}`
#   assumes a destination can be written repeatedly only when every write agrees.
#   That assumption is false in this variant, so the write must become a
#   two-phase "collect contenders -> choose nearest winner".
# - The original generator assumes destination uniqueness when selecting marker
#   pairs. That assumption must be removed, but only in a controlled way.
# d) Placement legality
# - T shapes must still stay disjoint and non-touching so color-1 components do
#   not merge and colored T branches stay singleton.
# - The blocker scrap used to create a farther marker is bundled into the same
#   occupancy shape as its T, so later controls cannot overlap it or touch it.
# - The scrap is embedded inside the destination region, is non-rectangular, and
#   is small enough to be absorbed by the existing small-fragment logic.
# e) Visual observability
# - The rule depends only on visible geometry: how far the target starts along
#   the stem direction. No color-id ordering is used.
# - The farther marker's extra scrap creates a literal gap of control cells on
#   the stem ray, so humans can compare the two competing Ts by eye.
# f) Visual cleanliness
# - Only one mandatory contested destination is introduced per grid. This keeps
#   the image readable while still making the new rule unavoidable.
# - Colored-center Ts are kept away from the contested group so the decisive cue
#   is spatial distance rather than a mix of orthogonal rule changes.
# g) Color-change inference
# - Normal Ts in the contested group come from different source regions, so the
#   competing replacement colors are visibly different.
# - A tie between equal distances would be hard to infer from 2-4 examples, so
#   the generator avoids such cases. The solver only accepts equal-distance ties
#   when the tied replacements agree.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


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
        if marker["kind"] not in {"normal", "colored"}:
            marker["source"] = dsu.find(marker["source_comp"])
            continue
        dest = dsu.find(marker["dest_comp"])
        source = dsu.find(marker["source_comp"])
        marker["dest"] = dest
        marker["source"] = source
        marker["replacement"] = (
            int(marker["target_color"]) if marker["kind"] == "colored" else int(root_color[source])
        )
        contests.setdefault(dest, []).append(marker)

    for dest, group in contests.items():
        best_distance = min(int(marker["dest_distance"]) for marker in group)
        winners = [marker for marker in group if int(marker["dest_distance"]) == best_distance]
        winner_colors = {int(marker["replacement"]) for marker in winners}
        assert len(winner_colors) == 1, "equidistant T controls for one region must agree"
        final_color[dest] = int(next(iter(winner_colors)))

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
    Repaint regions from T-shaped controls, erase the controls, and optionally rotate if a color-9 L marker is present.

    Input structure:
    - The grid is tiled by connected monochrome regions, plus control shapes written on top of them.
    - Color 1 forms normal T markers, while some T markers use a colored center with three singleton color-1 branches.
    - Several T markers may aim at the same destination region, and tiny non-rectangular scraps may appear as extra control blockers.
    - An optional 5-cell L of color 9 indicates a 90-degree rotation after recoloring.

    Rule:
    - The T stem points toward the region being targeted, and the two diagonals just behind the stem identify the source-side region.
    - A normal T repaints the destination to the source region's original color; a colored-center T repaints it to the center color.
    - If multiple T markers target the same region, compare how many stem steps are needed to reach that region after skipping control cells; the nearest T wins.
    - Every control cell is then filled with the final color of the region it belongs to, and the color-9 L marker rotates the finished output: ┐ means CCW 90 degrees, ┌ means CW 90 degrees.

    Key insight:
    - The diagonals still say where a T comes from, but under this variant the decisive cue for shared targets is the visible stem-to-region distance, not marker count alone.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (many_to_one_t) ---------------

# ── Tunable constants ──────────────────────────────────────────────────
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [2, 3, 3, 2, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 2

REGION_COLOR_POOL = [c for c in range(10) if c != 1]
COLORED_CENTER_POOL = [c for c in range(10) if c not in {1, 9}]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 300


def _get_bar(stem):
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center, stem):
    r, c = center
    bar = _get_bar(stem)
    return {
        (r, c),
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_branch_cells(center, stem):
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


def _choose_marker_counts() -> tuple[int, int, int]:
    n_markers = random.randint(MARKER_MIN, MARKER_MAX)
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
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == src
                    for bd in bar
                ):
                    continue
                step2 = (r + 2 * stem[0], c + 2 * stem[1])
                if step2 in occupied_controls:
                    continue
                if not _in_bounds(region.shape, step2[0], step2[1]):
                    continue
                if int(region[step2]) != dst:
                    continue
                candidates.append(
                    {
                        "src": int(src),
                        "dst": int(dst),
                        "center": (int(r), int(c)),
                        "stem": stem,
                        "distance": 2,
                        "shape": shape,
                        "buffer": _shape_neighborhood(shape),
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
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == src
                    for bd in bar
                ):
                    continue
                step2 = (r + 2 * stem[0], c + 2 * stem[1])
                step3 = (r + 3 * stem[0], c + 3 * stem[1])
                step4 = (r + 4 * stem[0], c + 4 * stem[1])
                if not all(_in_bounds(region.shape, rr, cc) for rr, cc in [step2, step3, step4]):
                    continue
                if not all(int(region[rr, cc]) == dst for rr, cc in [step2, step3, step4]):
                    continue
                if step4 in occupied_controls:
                    continue
                for side in bar:
                    side_cell = (step2[0] + side[0], step2[1] + side[1])
                    if not _in_bounds(region.shape, side_cell[0], side_cell[1]):
                        continue
                    if int(region[side_cell]) != dst:
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
                            "buffer": _shape_neighborhood(shape),
                            "scrap_cells": scrap_cells,
                        }
                    )
    return candidates


def _candidates_compatible(a: dict, b: dict) -> bool:
    if a["shape"] & b["shape"]:
        return False
    if a["shape"] & b["buffer"]:
        return False
    if b["shape"] & a["buffer"]:
        return False
    return True


def _select_shared_target_pair(region: np.ndarray, H: int, W: int) -> tuple[dict, dict] | None:
    direct_by_dst: dict[int, list[dict]] = {}
    blocked_by_dst: dict[int, list[dict]] = {}
    for src, dst in _find_adjacent_pairs(region):
        direct = _find_valid_direct_placements(region, src, dst, set(), set(), H, W)
        blocked = _find_valid_blocked_placements(region, src, dst, set(), set(), H, W)
        if direct:
            direct_by_dst.setdefault(int(dst), []).extend(direct)
        if blocked:
            blocked_by_dst.setdefault(int(dst), []).extend(blocked)

    shared_dsts = [dst for dst in direct_by_dst if dst in blocked_by_dst]
    random.shuffle(shared_dsts)
    for dst in shared_dsts:
        direct = direct_by_dst[dst][:]
        blocked = blocked_by_dst[dst][:]
        random.shuffle(direct)
        random.shuffle(blocked)
        for near in direct:
            for far in blocked:
                if int(near["src"]) == int(far["src"]):
                    continue
                if not _candidates_compatible(near, far):
                    continue
                return near, far
    return None


def _choose_remaining_pairs(
    directed_pairs: list[tuple[int, int]],
    shared_dst: int,
    count: int,
) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    pairs = directed_pairs[:]
    random.shuffle(pairs)
    chosen: list[tuple[int, int]] = []
    used_dsts: set[int] = set()
    for src, dst in pairs:
        if dst == shared_dst:
            continue
        if dst not in used_dsts:
            chosen.append((int(src), int(dst)))
            used_dsts.add(int(dst))
            if len(chosen) >= count:
                return chosen
    if not chosen:
        fallback = [pair for pair in pairs if pair[1] != shared_dst]
        if not fallback:
            return []
        chosen.append(tuple(map(int, random.choice(fallback))))
    while len(chosen) < count:
        chosen.append(tuple(map(int, random.choice(chosen))))
    return chosen


def _choose_colored_targets(
    specs: list[dict], input_colors: list[int], reserved_color: int
) -> bool:
    for spec in specs:
        if spec["kind"] != "colored":
            spec["target_color"] = None
            continue
        dst = int(spec["dst"])
        cands = [
            c
            for c in COLORED_CENTER_POOL
            if c != int(input_colors[dst]) and c != int(reserved_color)
        ]
        if not cands:
            return False
        spec["target_color"] = int(random.choice(cands))
    return True


def _build_specs(
    region: np.ndarray, shared_near: dict, shared_far: dict, n_markers: int, n_colored: int
) -> list[dict] | None:
    directed_pairs = _find_adjacent_pairs(region)
    remaining_pairs = _choose_remaining_pairs(directed_pairs, int(shared_near["dst"]), n_markers - 2)
    if len(remaining_pairs) != n_markers - 2:
        return None

    specs = [
        {"kind": "normal", "src": int(shared_near["src"]), "dst": int(shared_near["dst"]), "forced": shared_near},
        {"kind": "normal", "src": int(shared_far["src"]), "dst": int(shared_far["dst"]), "forced": shared_far},
    ]
    for src, dst in remaining_pairs:
        specs.append({"kind": "normal", "src": int(src), "dst": int(dst), "forced": None})

    candidate_indices = []
    seen_dsts: set[int] = set()
    for idx, spec in enumerate(specs[2:], start=2):
        dst = int(spec["dst"])
        if dst in seen_dsts:
            continue
        candidate_indices.append(idx)
        seen_dsts.add(dst)
    if len(candidate_indices) < n_colored:
        return None
    for idx in random.sample(candidate_indices, n_colored):
        specs[idx]["kind"] = "colored"
    return specs


def _place_specs(region: np.ndarray, specs: list[dict], H: int, W: int) -> tuple[list[dict], list[dict]] | None:
    occupied_controls: set[tuple[int, int]] = set()
    occupied_buffer: set[tuple[int, int]] = set()
    placed_markers: list[dict] = []
    placed_scraps: list[dict] = []

    forced_specs = [spec for spec in specs if spec["forced"] is not None]
    normal_specs = [spec for spec in specs if spec["forced"] is None]
    random.shuffle(normal_specs)
    placement_order = forced_specs + normal_specs

    for spec in placement_order:
        forced = spec["forced"]
        if forced is not None:
            shape = set(forced["shape"])
            if not _shape_fits(shape, occupied_controls, occupied_buffer, H, W):
                return None
            candidate = forced
        else:
            candidates = _find_valid_direct_placements(
                region, int(spec["src"]), int(spec["dst"]), occupied_controls, occupied_buffer, H, W
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
                "target_color": spec.get("target_color"),
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
    lut = np.array(colors, dtype=int)
    return lut[region]


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


def _winner_map(placed_markers: list[dict], input_colors: list[int]) -> dict[int, int] | None:
    grouped: dict[int, list[dict]] = {}
    for marker in placed_markers:
        grouped.setdefault(int(marker["dst"]), []).append(marker)
    winners: dict[int, int] = {}
    for dst, group in grouped.items():
        best_distance = min(int(marker["distance"]) for marker in group)
        best = [marker for marker in group if int(marker["distance"]) == best_distance]
        replacements = set()
        for marker in best:
            if marker["kind"] == "colored":
                replacements.add(int(marker["target_color"]))
            else:
                replacements.add(int(input_colors[int(marker["src"])]))
        if len(replacements) != 1:
            return None
        winners[int(dst)] = int(next(iter(replacements)))
    return winners


def _try_place_l9(
    region: np.ndarray,
    input_colors: list[int],
    occupied_controls: set[tuple[int, int]],
    occupied_buffer: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
    allow_l9: bool,
) -> str | None:
    if (not allow_l9) or random.random() >= L9_PROBABILITY:
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
            if int(input_colors[rid]) == 9:
                continue
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            candidates.append((cells, rid))

    if not candidates:
        return None
    cells, rid = random.choice(candidates)
    for cr, cc in cells:
        input_grid[cr, cc] = 9
        output_grid[cr, cc] = out_colors[rid]
    return orient


def _control_occupancy_from_placements(
    placed_markers: list[dict], placed_scraps: list[dict]
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    occupied_controls: set[tuple[int, int]] = set()
    occupied_buffer: set[tuple[int, int]] = set()
    for marker in placed_markers:
        shape = _t_cells(marker["center"], marker["stem"])
        _register_control_shape(shape, occupied_controls, occupied_buffer)
    for scrap in placed_scraps:
        _register_control_shape(set(scrap["cells"]), occupied_controls, occupied_buffer)
    return occupied_controls, occupied_buffer


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    shared_pair = _select_shared_target_pair(region, H, W)
    if shared_pair is None:
        return None
    shared_near, shared_far = shared_pair

    n_markers, _n_normal, n_colored = _choose_marker_counts()
    specs = _build_specs(region, shared_near, shared_far, n_markers, n_colored)
    if specs is None:
        return None

    pool = list(REGION_COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    spare_colors = [c for c in REGION_COLOR_POOL if c not in input_colors]
    if not spare_colors:
        return None
    non9_spares = [c for c in spare_colors if c != 9]
    scrap_color = int(random.choice(non9_spares if non9_spares else spare_colors))
    allow_l9 = scrap_color != 9

    if not _choose_colored_targets(specs, input_colors, scrap_color):
        return None

    placed = _place_specs(region, specs, H, W)
    if placed is None:
        return None
    placed_markers, placed_scraps = placed

    winners = _winner_map(placed_markers, input_colors)
    if winners is None:
        return None

    shared_dst = int(shared_near["dst"])
    shared_group = [m for m in placed_markers if int(m["dst"]) == shared_dst]
    if len(shared_group) < 2:
        return None
    distances = sorted(int(m["distance"]) for m in shared_group)
    if distances[0] == distances[1]:
        return None

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
    l9_orient = _try_place_l9(
        region,
        input_colors,
        occupied_controls,
        occupied_buffer,
        input_grid,
        output_grid,
        out_colors,
        allow_l9,
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
