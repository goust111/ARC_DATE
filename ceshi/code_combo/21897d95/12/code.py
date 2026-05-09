# Combo variant of 21897d95 (index 12)
# combo_spec: D5/0+D10/2
# Name: 图例双阶段处理
#
# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior and traced execution
# - Pixel form:
#   - every T control is now the same 4-cell color-1 tetromino;
#   - there are no colored-center T markers;
#   - a compact 2xN legend on the top edge lists `(source_color, replacement_color)`
#     pairs, column by column;
#   - the task runs in two recolor stages.
# - Stage trace on one mapped marker:
#   - original 21897d95: a normal T copies its source color directly to the target,
#     while a colored T overrides its target immediately with its center color.
#   - this combo: first every T behaves like the original normal T, so the target
#     region becomes the source-side color; then only markers whose source color
#     appears in the legend act again, repainting their current destination component
#     to the legend replacement.
# - Why the second stage is genuinely sequential here:
#   - the generator creates one mapped T whose stage-1 destination becomes adjacent to
#     a separate region that already has the same color;
#   - by stage 2 those two pieces are one connected component, so the mapped T
#     recolors both together;
#   - a flat one-stage "legend directly sets each destination" baseline would recolor
#     only the original destination, not the newly merged buddy region.
#
# b) Complexity calibration
# - The legend width is limited to 2-4 columns. That is enough repetition for humans
#   to infer the source→replacement table without overcrowding the edge.
# - Only one color duplication is forced structurally: the mapped source region shares
#   its input color with one passive buddy region. This is the minimum extra freedom
#   needed to make the second-stage component expansion visible.
# - The generator still keeps all T markers in a single visual family, so there is one
#   extra decoding step (read the legend) and one extra execution step (second-stage
#   repaint), but not many independent symbol grammars.
#
# c) Implicit-assumption audit against the original solver
# - The original scan for colored-center T markers is invalid now because non-1 legend
#   cells on the edge would be misclassified. The combo parser accepts only pure color-1
#   tetromino T markers and consumes the legend before small-scrap detection.
# - The original simultaneous `final_color[dest] = source_or_center_color` model is no
#   longer correct. We must explicitly build a stage-1 intermediate grid and then
#   re-label connected components for stage 2.
# - The original assumption that all region colors are distinct is no longer safe in the
#   generator: one passive buddy region intentionally shares a source color so that the
#   stage-2 destination component can expand.
# - Legend cells must be part of the control mask for both stages; otherwise they would
#   leak into component labeling as tiny bogus regions.
#
# d) Placement legality
# - The top-edge legend occupies a 2xN strip inside one owner region; row 2 and the
#   left/right side cells remain owner-colored so the legend is uniquely decodable.
# - T markers are buffered away from the legend and from each other with the same
#   neighborhood exclusion used in the reference variants.
# - The passive buddy region is never used as a T destination, preserving the intended
#   stage-2 expansion witness.
# - The optional L9 marker still obeys the original no-overlap constraints and is erased
#   back into its owner region only after its orientation is read.
#
# e) Visual observability
# - Humans can see that all T markers share the same shape, so the extra information
#   must come from the legend.
# - The legend is reusable across multiple T markers, giving repeated evidence for the
#   color mapping.
# - The generator enforces that the combo output differs from a direct one-stage legend
#   baseline by at least one whole region-sized chunk, so the second stage is not a
#   hidden no-op.
#
# f) Visual cleanliness
# - The legend stays in one compact edge strip.
# - Only one duplicated base color is deliberately introduced, and it is tied to a
#   single passive buddy region. This keeps the mosaic readable while still making the
#   stage-2 merge effect visible.
# - Control symbols are separated by at least one cell of buffer, so T shapes and the
#   L marker remain visually parseable.
#
# g) Change inferability
# - Legend keys are source colors that actually appear in T source regions, and the
#   generator guarantees at least one mapped source color is used by a marker.
# - Replacement colors are chosen to be different from their source colors, so the
#   legend has a visible effect.
# - The sequential witness is quantified during generation by comparing the combo solver
#   against a flat one-stage legend baseline on the exact same input.

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


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if int(stem[0]) == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    bar = _get_bar(stem)
    return {
        (int(r), int(c)),
        (int(r + stem[0]), int(c + stem[1])),
        (int(r + bar[0][0]), int(c + bar[0][1])),
        (int(r + bar[1][0]), int(c + bar[1][1])),
    }


def _t_neighborhood(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _t_cells(center, stem)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (int(r + dr), int(c + dc))
            if nb not in cells:
                out.add(nb)
    return out


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
        labels, n_labels = label_mask((g == int(col)) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == int(k))
            if coords.size == 0:
                continue
            for rr, cc in coords:
                comp_id[int(rr), int(cc)] = cid
            comp_colors.append(int(col))
            comp_areas.append(int(len(coords)))
            cid += 1
    return comp_id, comp_colors, comp_areas


def _mark_cells(mask: np.ndarray, markers: list[dict]) -> None:
    for marker in markers:
        for r, c in marker["cells"]:
            mask[int(r), int(c)] = True


def _adjacent_component_counts(cells: set[tuple[int, int]], comp_id: np.ndarray) -> Counter:
    counts: Counter = Counter()
    for r, c in cells:
        for dr, dc in DIRS:
            rr, cc = int(r + dr), int(c + dc)
            if _in_bounds(comp_id.shape, rr, cc) and int(comp_id[rr, cc]) >= 0:
                counts[int(comp_id[rr, cc])] += 1
    return counts


def _first_component_from(
    center: tuple[int, int], direction: tuple[int, int], comp_id: np.ndarray, start: int
) -> int | None:
    r, c = center
    step = int(start)
    while True:
        rr = int(r + direction[0] * step)
        cc = int(c + direction[1] * step)
        if not _in_bounds(comp_id.shape, rr, cc):
            return None
        if int(comp_id[rr, cc]) >= 0:
            return int(comp_id[rr, cc])
        step += 1


def _find_top_legend(g: np.ndarray) -> dict:
    H, W = g.shape
    assert H >= 3 and W >= 4, "legend variant requires at least 3 rows and 4 columns"
    candidates: list[dict] = []
    max_width = min(4, W - 2)
    next_id = 1_000_000
    for width in range(2, max_width + 1):
        for c0 in range(1, W - width):
            c1 = c0 + width - 1
            owner = int(g[2, c0])
            if any(int(g[2, c]) != owner for c in range(c0, c1 + 1)):
                continue
            if int(g[0, c0 - 1]) != owner or int(g[1, c0 - 1]) != owner:
                continue
            if int(g[0, c1 + 1]) != owner or int(g[1, c1 + 1]) != owner:
                continue
            pairs: list[tuple[int, int]] = []
            ok = True
            for c in range(c0, c1 + 1):
                top = int(g[0, c])
                bottom = int(g[1, c])
                if top in {1, 9} or bottom in {1, 9}:
                    ok = False
                    break
                if top == owner or bottom == owner:
                    ok = False
                    break
                pairs.append((top, bottom))
            if not ok:
                continue
            legend = {
                "mid": next_id,
                "kind": "legend",
                "cells": {(r, c) for r in range(2) for c in range(c0, c1 + 1)},
                "pairs": pairs,
                "mapping": {src: dst for src, dst in pairs},
            }
            if len(legend["mapping"]) != len(legend["pairs"]):
                continue
            candidates.append(legend)
            next_id += 1
    assert len(candidates) == 1, "expected exactly one top-edge legend"
    return candidates[0]


def _parse_controls(g: np.ndarray) -> tuple[list[dict], dict, list[dict]]:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        int(k): {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == int(k))}
        for k in range(1, int(n_1) + 1)
    }

    markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)
    next_mid = 0

    for cells in one_components.values():
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        marker = {
            "mid": next_mid,
            "kind": "normal",
            "center": center,
            "stem": stem,
            "bar": bar,
            "cells": set(cells),
        }
        next_mid += 1
        markers.append(marker)
        used_ones |= cells
        _mark_cells(control, [marker])

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a normal T marker"

    legend = _find_top_legend(g)
    _mark_cells(control, [legend])

    small_markers: list[dict] = []
    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == int(col)) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == int(k))
            if len(coords) > 5:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if len(coords) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
                continue
            marker = {
                "mid": next_mid,
                "kind": "small",
                "cells": {tuple(map(int, xy)) for xy in coords},
            }
            next_mid += 1
            small_markers.append(marker)
            _mark_cells(control, [marker])

    return markers, legend, small_markers


def _apply_controls(
    g: np.ndarray,
    current_t_markers: list[dict],
    active_t_markers: list[dict],
    extra_markers: list[dict],
    fill_extra: bool,
    mode: str,
    legend_map: dict[int, int],
) -> np.ndarray:
    control = np.zeros(g.shape, dtype=bool)
    _mark_cells(control, current_t_markers)
    _mark_cells(control, extra_markers)

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"

    dsu = _DSU(len(comp_colors))
    active_ids = {int(marker["mid"]) for marker in active_t_markers}
    marker_source_comp: dict[int, int] = {}
    marker_dest_comp: dict[int, int] = {}

    for marker in current_t_markers:
        center = marker["center"]
        stem = marker["stem"]
        dest = _first_component_from(center, stem, comp_id, start=2)
        assert dest is not None, "T marker must point at a destination region"

        source_parts: list[int] = []
        for bar_dir in marker["bar"]:
            rr = int(center[0] + stem[0] + bar_dir[0])
            cc = int(center[1] + stem[1] + bar_dir[1])
            assert _in_bounds(g.shape, rr, cc) and int(comp_id[rr, cc]) >= 0
            source_parts.append(int(comp_id[rr, cc]))

        source = source_parts[0]
        for part in source_parts[1:]:
            assert comp_colors[part] == comp_colors[source], "T source diagonals must share a color"
            dsu.union(source, part)

        for comp in _adjacent_component_counts(marker["cells"], comp_id):
            if comp_colors[comp] == comp_colors[source]:
                dsu.union(source, comp)

        marker_source_comp[int(marker["mid"])] = source
        marker_dest_comp[int(marker["mid"])] = int(dest)

    extra_owner_comp: dict[int, int] = {}
    for marker in extra_markers:
        counts = _adjacent_component_counts(marker["cells"], comp_id)
        assert counts, f"{marker['kind']} marker must touch its owner region"
        owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
        for comp in counts:
            if comp_colors[comp] == comp_colors[owner]:
                dsu.union(owner, comp)
        extra_owner_comp[int(marker["mid"])] = int(owner)

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    for marker in current_t_markers:
        source = dsu.find(marker_source_comp[int(marker["mid"])])
        marker["source"] = source
        marker["source_color"] = int(root_color[source])

    final_color = dict(root_color)
    for marker in active_t_markers:
        source = dsu.find(marker_source_comp[int(marker["mid"])])
        dest = dsu.find(marker_dest_comp[int(marker["mid"])])
        if mode == "direct":
            replacement = int(root_color[source])
        elif mode == "legend":
            replacement = int(marker["legend_replacement"])
        elif mode == "flat":
            replacement = int(legend_map.get(int(root_color[source]), int(root_color[source])))
        else:
            raise AssertionError(f"unknown mode: {mode}")
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting T controls for one region"
        final_color[dest] = replacement

    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[dsu.find(cid)])

    for marker in current_t_markers:
        color = int(final_color[marker["source"]])
        for r, c in marker["cells"]:
            out[int(r), int(c)] = color

    if fill_extra:
        for marker in extra_markers:
            owner = dsu.find(extra_owner_comp[int(marker["mid"])])
            color = int(final_color[owner])
            for r, c in marker["cells"]:
                out[int(r), int(c)] = color

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def _solve_unrotated_combo(g: np.ndarray) -> np.ndarray:
    markers, legend, small_markers = _parse_controls(g)
    extras = [legend] + small_markers
    mid = _apply_controls(g, markers, markers, extras, fill_extra=False, mode="direct", legend_map=legend["mapping"])
    active = []
    for marker in markers:
        src_color = int(marker["source_color"])
        if src_color in legend["mapping"]:
            marker["legend_replacement"] = int(legend["mapping"][src_color])
            active.append(marker)
    return _apply_controls(mid, markers, active, extras, fill_extra=True, mode="legend", legend_map=legend["mapping"])


def _solve_unrotated_flat(g: np.ndarray) -> np.ndarray:
    markers, legend, small_markers = _parse_controls(g)
    extras = [legend] + small_markers
    return _apply_controls(g, markers, markers, extras, fill_extra=True, mode="flat", legend_map=legend["mapping"])


def _find_l9_orientation(g: np.ndarray) -> str | None:
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    for k in range(1, int(n) + 1):
        coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == int(k))}
        if len(coords) != 5:
            continue
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        for cr, cc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
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
    Repaint regions from uniform T markers in two stages: first copy source colors directly, then use an edge legend to override selected stage-1 targets, with an optional L-marker rotation at the end.

    Input structure:
    - The grid is an irregular mosaic of monochrome regions, plus control symbols on top of that mosaic.
    - Every T control is a 4-cell color-1 tetromino; there are no colored-center T markers in this combo.
    - A compact 2xN legend sits on the top edge. In each column, the top cell is a source color and the bottom cell is its stage-2 replacement color.
    - An optional 5-cell color-9 L marker may be embedded inside one region and still indicates a quarter-turn of the final output.

    Rule:
    - For every T, the single stem points toward the destination region, while the two diagonals behind the stem identify the source side.
    - Stage 1 applies all T markers simultaneously as ordinary copy-controls: each destination becomes the source side's current color, and the T cells themselves are filled with that source-side color.
    - Stage 2 re-evaluates the stage-1 grid. If a T marker's source color appears in the legend, its current destination component is repainted to the legend's replacement color. Because this happens after stage 1, that destination may already have merged with adjacent same-colored regions.
    - The legend strip, any small embedded scraps, and the optional L marker are erased into their touching owner region after the recoloring is complete. L orientation remains ┐ → CCW 90° and ┌ → CW 90°.

    Key insight:
    - The T geometry still says who acts on whom, but the final color is not always decided in one step. First the T copies a visible source color; only afterward does the legend decide which of those copied colors trigger a second repaint, and that second repaint can spread over a larger component than the original target.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated_combo(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (legend two-stage) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.6
REGION_COUNTS = [5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 2, 1]

MARKER_MIN = 4
MARKER_MAX = 7
LEGEND_MIN = 2
LEGEND_MAX = 4

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 500
PAIR_RETRIES = 200
PLACEMENT_RETRIES = 20
MIN_SEQ_DELTA = MIN_REGION_CELLS


def _legend_cells(start_col: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(2) for c in range(start_col, start_col + width)}


def _legend_neighborhood(start_col: int, width: int, H: int, W: int) -> set[tuple[int, int]]:
    cells = _legend_cells(start_col, width)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in cells:
                out.add((rr, cc))
    return out


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    region = np.full((H, W), -1, dtype=int)
    seeds: set[tuple[int, int]] = set()
    for _ in range(SEED_ATTEMPTS):
        if len(seeds) >= n_regions:
            break
        seeds.add((random.randint(1, H - 2), random.randint(1, W - 2)))
    if len(seeds) < n_regions:
        return None

    frontier: list[tuple[int, int, int]] = []
    for i, (r, c) in enumerate(list(seeds)):
        region[r, c] = i
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and int(region[nr, nc]) == -1:
                frontier.append((nr, nc, i))
    random.shuffle(frontier)

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if int(region[r, c]) != -1:
            continue
        region[r, c] = rid
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and int(region[nr, nc]) == -1:
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
    undirected: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            for dr, dc in DIRS:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < H and 0 <= nc < W:
                    a, b = int(region[r, c]), int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed: list[tuple[int, int]] = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _region_neighbors(region: np.ndarray) -> dict[int, set[int]]:
    neighbors = {rid: set() for rid in range(int(region.max()) + 1)}
    for a, b in _find_adjacent_pairs(region):
        neighbors[int(a)].add(int(b))
    return neighbors


def _find_legend_slots(region: np.ndarray) -> list[tuple[int, int, int]]:
    H, W = region.shape
    slots = []
    if H < 3 or W < 4:
        return slots
    for width in range(LEGEND_MIN, min(LEGEND_MAX, W - 2) + 1):
        for c0 in range(1, W - width):
            c1 = c0 + width - 1
            owner = int(region[0, c0])
            if any(int(region[r, c]) != owner for r in range(3) for c in range(c0, c1 + 1)):
                continue
            if any(int(region[r, c0 - 1]) != owner for r in range(3)):
                continue
            if any(int(region[r, c1 + 1]) != owner for r in range(3)):
                continue
            slots.append((owner, c0, width))
    return slots


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    H, W = region.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in DIRS:
                bar = _get_bar(stem)
                cells = _t_cells((r, c), stem)
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
                found = False
                while True:
                    nr, nc = int(r + stem[0] * step), int(c + stem[1] * step)
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied_cells or (nr, nc) in cells:
                        step += 1
                        continue
                    found = int(region[nr, nc]) == int(dst)
                    break
                if found:
                    candidates.append(((int(r), int(c)), stem))
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


def _place_markers(
    region: np.ndarray,
    marker_defs: list[tuple[int, int]],
    initial_occupied_cells: set[tuple[int, int]],
    initial_occupied_neighbors: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        occupied_cells = set(initial_occupied_cells)
        occupied_neighbors = set(initial_occupied_neighbors)
        pending = list(marker_defs)
        placed: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []

        while pending:
            options = []
            dead = False
            for idx, (src, dst) in enumerate(pending):
                cands = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors)
                if not cands:
                    dead = True
                    break
                options.append((len(cands), random.random(), idx, cands))
            if dead:
                break

            _, _, pick_idx, cands = min(options, key=lambda item: (item[0], item[1]))
            src, dst = pending.pop(int(pick_idx))
            center, stem = random.choice(cands)
            _register_marker(center, stem, occupied_cells, occupied_neighbors)
            placed.append((center, stem, int(src), int(dst)))

        if not pending:
            return placed
    return None


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_legend_input(grid: np.ndarray, start_col: int, pairs: list[tuple[int, int]]) -> None:
    for idx, (src_color, dst_color) in enumerate(pairs):
        c = start_col + idx
        grid[0, c] = int(src_color)
        grid[1, c] = int(dst_color)


def _stamp_markers_input(
    grid: np.ndarray,
    placed_markers: list[tuple[tuple[int, int], tuple[int, int], int, int]],
) -> None:
    for center, stem, _src, _dst in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[int(cr), int(cc)] = 1


def _try_place_l9(
    region: np.ndarray,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    input_grid: np.ndarray,
) -> None:
    if random.random() >= L9_PROBABILITY:
        return
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            cells = [(int(r + dr), int(c + dc)) for dr, dc in offsets]
            if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if any((cr, cc) in occupied_cells for cr, cc in cells):
                continue
            if any((cr, cc) in occupied_neighbors for cr, cc in cells):
                continue
            candidates.append((int(r), int(c)))

    if not candidates:
        return
    lr, lc = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[int(lr + dr), int(lc + dc)] = 9


def _choose_interaction(
    neighbors: dict[int, set[int]],
    owner_rid: int,
) -> tuple[int, int, int] | None:
    candidates: list[tuple[int, int, int]] = []
    for src, dsts in neighbors.items():
        if src == owner_rid:
            continue
        for dst in dsts:
            if dst == owner_rid:
                continue
            for buddy in neighbors[dst]:
                if buddy in {src, dst, owner_rid}:
                    continue
                candidates.append((int(src), int(dst), int(buddy)))
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _choose_marker_pairs(
    directed_pairs: list[tuple[int, int]],
    fixed_pair: tuple[int, int],
    required_sources: list[int],
    forbidden_sources: set[int],
    forbidden_dests: set[int],
) -> list[tuple[int, int]] | None:
    base_candidates = [
        (int(src), int(dst))
        for src, dst in directed_pairs
        if int(dst) not in forbidden_dests and int(src) not in forbidden_sources
    ]
    by_source: dict[int, list[tuple[int, int]]] = {}
    for src, dst in base_candidates:
        by_source.setdefault(int(src), []).append((int(src), int(dst)))

    min_markers = max(MARKER_MIN, 1 + len(required_sources))
    feasible_dests = {dst for _src, dst in base_candidates} - {int(fixed_pair[1])}
    max_markers = min(MARKER_MAX, 1 + len(feasible_dests))
    if max_markers < min_markers:
        return None

    for _ in range(PAIR_RETRIES):
        n_markers = random.randint(min_markers, max_markers)
        chosen = [tuple(map(int, fixed_pair))]
        used_dests = {int(fixed_pair[1])}
        ok = True

        reqs = list(required_sources)
        random.shuffle(reqs)
        for src in reqs:
            opts = [pair for pair in by_source.get(int(src), []) if int(pair[1]) not in used_dests]
            if not opts:
                ok = False
                break
            pair = random.choice(opts)
            chosen.append(pair)
            used_dests.add(int(pair[1]))
        if not ok:
            continue

        remaining = [pair for pair in base_candidates if int(pair[1]) not in used_dests]
        random.shuffle(remaining)
        for pair in remaining:
            if len(chosen) >= n_markers:
                break
            chosen.append(pair)
            used_dests.add(int(pair[1]))

        if len(chosen) == n_markers:
            random.shuffle(chosen)
            return chosen
    return None


def _choose_input_colors(
    n_regions: int,
    src_rid: int,
    buddy_rid: int,
) -> list[int] | None:
    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions - 1 > len(pool):
        return None
    colors = [0] * n_regions
    shared = pool[0]
    others = [c for c in pool if c != shared]
    if len(others) < n_regions - 2:
        return None
    assign = others[: n_regions - 2]
    idx = 0
    for rid in range(n_regions):
        if rid in {src_rid, buddy_rid}:
            colors[rid] = shared
        else:
            colors[rid] = assign[idx]
            idx += 1
    return colors


def _choose_replacement_map(
    source_colors: list[int],
    owner_color: int,
) -> dict[int, int] | None:
    unique_sources = list(dict.fromkeys(source_colors))
    available = [c for c in COLOR_POOL if c != owner_color]
    for _ in range(PAIR_RETRIES):
        random.shuffle(available)
        repls: dict[int, int] = {}
        used_targets: set[int] = set()
        ok = True
        for src in unique_sources:
            choices = [c for c in available if c != src and c not in used_targets]
            if not choices:
                ok = False
                break
            target = random.choice(choices)
            repls[int(src)] = int(target)
            used_targets.add(int(target))
        if ok:
            return repls
    return None


def _combo_delta_vs_flat(g: np.ndarray) -> int:
    combo = _solve_unrotated_combo(g)
    flat = _solve_unrotated_flat(g)
    return int(np.count_nonzero(combo != flat))


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(10, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    legend_slots = _find_legend_slots(region)
    if not legend_slots:
        return None

    neighbors = _region_neighbors(region)
    directed_pairs = _find_adjacent_pairs(region)

    random.shuffle(legend_slots)
    for owner_rid, start_col, width in legend_slots:
        interaction = _choose_interaction(neighbors, owner_rid)
        if interaction is None:
            continue
        src_rid, mapped_dst, buddy_rid = interaction

        input_colors = _choose_input_colors(n_regions, src_rid, buddy_rid)
        if input_colors is None:
            continue

        candidate_sources = [
            rid
            for rid in range(n_regions)
            if rid not in {owner_rid, buddy_rid, src_rid}
            and input_colors[rid] != input_colors[src_rid]
        ]
        if len(candidate_sources) < width - 1:
            continue
        extra_legend_sources = random.sample(candidate_sources, width - 1)
        legend_source_rids = [src_rid] + extra_legend_sources
        legend_source_colors = [input_colors[rid] for rid in legend_source_rids]
        replacement_map = _choose_replacement_map(legend_source_colors, input_colors[owner_rid])
        if replacement_map is None:
            continue
        legend_pairs = [(int(color), int(replacement_map[color])) for color in legend_source_colors]

        fixed_pair = (int(src_rid), int(mapped_dst))
        marker_pairs = _choose_marker_pairs(
            directed_pairs,
            fixed_pair=fixed_pair,
            required_sources=extra_legend_sources,
            forbidden_sources={int(buddy_rid)},
            forbidden_dests={int(owner_rid), int(buddy_rid)},
        )
        if marker_pairs is None:
            continue

        occupied_cells = set(_legend_cells(start_col, width))
        occupied_neighbors = _legend_neighborhood(start_col, width, H, W)
        placed_markers = _place_markers(region, marker_pairs, occupied_cells, occupied_neighbors)
        if placed_markers is None:
            continue

        for center, stem, _src, _dst in placed_markers:
            occupied_cells |= _t_cells(center, stem)
            occupied_neighbors |= (_t_neighborhood(center, stem) - _t_cells(center, stem))

        input_grid = _build_color_map(region, input_colors)
        _stamp_legend_input(input_grid, start_col, legend_pairs)
        _stamp_markers_input(input_grid, placed_markers)

        try:
            seq_delta = _combo_delta_vs_flat(input_grid)
        except Exception:
            continue
        if seq_delta < MIN_SEQ_DELTA:
            continue

        _try_place_l9(region, occupied_cells, occupied_neighbors, input_grid)

        try:
            _ = main(input_grid)
        except Exception:
            continue
        return input_grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
