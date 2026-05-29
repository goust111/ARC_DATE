# Variant of 21897d95
# Dimension: D18 — 边界处T标记的对角线源区域检测 (meta)
# Change: 茎方向+横杆方向的两个对角线单元必须都在源区域中，且两个对角线单元颜色必须一致... → 两个对角线可以在不同区域中（双源）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the D18 variant
# -----------------------------------
# a) Concrete behavior and trace
#    - The normal color-1 T still sits on a boundary and still points with its stem
#      toward the destination region.
#    - The two diagnostic cells are still read at center + stem + each bar direction.
#      In the original task both had to land in the same-color source side, so the
#      destination finished as one solid color.
#    - In this variant the two diagonals may sit in two different regions with two
#      different colors. Example: a downward T whose lower-left diagonal is color 4
#      and lower-right diagonal is color 7 now repaints the destination as horizontal
#      stripes: grid rows 1,3,5,... use 4 and rows 2,4,6,... use 7.
#    - The affected pipeline stage is the "replacement color" computation. Original:
#      source_parts -> assert same color -> one scalar replacement. Variant:
#      source_parts -> if same root/color keep scalar replacement, else build a
#      two-color stripe style and render the destination cell-by-cell by row parity.
# b) Complexity calibration
#    - If the generator still emitted mostly single-source Ts, the variant would
#      visually collapse back to the original task. To prevent that, generation emits
#      only dual-source normal T markers.
#    - Colored-center Ts are suppressed in generation. Mixing "literal center-color
#      override" with "dual-source stripe override" would create two competing
#      recoloring cues on one marker and make the new rule harder to infer.
# c) Hidden assumptions from the original main()
#    - `assert comp_colors[part] == comp_colors[source]` is no longer valid for normal
#      Ts; only same-color diagonals should merge, different-color diagonals must stay
#      separate.
#    - `final_color[dest]` as one integer is no longer sufficient. It becomes a style:
#      either `("solid", c)` or `("stripe", c_a, c_b)`.
#    - Marker erasure formerly copied one source-side final color. A dual-source T has
#      no single owner color, so its marker cells are filled with the same stripe style
#      as the destination. This is the smallest compatible extension.
# d) Placement legality
#    - The generator now searches directly for T placements whose two diagonal cells
#      belong to different regions and whose step-2 stem cell belongs to a third
#      destination region. Occupancy and neighborhood buffers from the original code
#      are preserved so markers do not overlap or merge accidentally.
# e) Visual observability
#    - Every generated puzzle contains at least four dual-source normal Ts, so the
#      striped effect is repeated and cannot be mistaken for noise.
#    - Stripe colors come directly from the visibly adjacent diagonal source regions;
#      no hidden ordering by color id is used.
# f) Visual neatness
#    - Horizontal stripes are chosen because the rule mentions odd/even rows and they
#      read cleanly inside irregular regions. Marker spacing and optional L9 placement
#      remain buffered to avoid clutter.
# g) Color-rule inferability
#    - Input region colors remain distinct, so each stripe pair can be traced back to
#      the two diagonal source regions by eye. The only row-dependent logic is absolute
#      row parity: 1st/3rd/5th rows use source A, 2nd/4th/6th use source B. Source A is
#      the first bar-side diagonal in geometric order: left then right for vertical
#      stems, up then down for horizontal stems.


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


def _style_at_row(style: tuple, r: int) -> int:
    if style[0] == "solid":
        return int(style[1])
    assert style[0] == "stripe"
    return int(style[1] if int(r) % 2 == 0 else style[2])


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

            if comp_colors[source_parts[0]] == comp_colors[source_parts[1]]:
                dsu.union(source_parts[0], source_parts[1])

            touching = adjacent_component_counts(marker["cells"])
            for part in source_parts:
                for comp in touching:
                    if comp_colors[comp] == comp_colors[part]:
                        dsu.union(part, comp)

            marker["dest_comp"] = dest
            marker["source_parts"] = source_parts
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

    final_style = {root: ("solid", int(col)) for root, col in root_color.items()}
    for marker in markers:
        if marker["kind"] == "small":
            marker["paint_mode"] = "root"
            marker["paint_root"] = dsu.find(marker["source_comp"])
            continue

        if marker["kind"] == "colored":
            source_roots = [dsu.find(part) for part in marker["source_parts"]]
            assert len(set(source_roots)) == 1, "colored T markers still require one source side"
            dest = dsu.find(marker["dest_comp"])
            replacement = ("solid", int(marker["target_color"]))
            assert final_style[dest] in {("solid", root_color[dest]), replacement}, "conflicting T controls for one region"
            final_style[dest] = replacement
            marker["paint_mode"] = "root"
            marker["paint_root"] = source_roots[0]
            continue

        dest = dsu.find(marker["dest_comp"])
        source_roots = [dsu.find(part) for part in marker["source_parts"]]
        source_colors = [int(root_color[root]) for root in source_roots]
        if len(set(source_roots)) == 1:
            replacement = ("solid", source_colors[0])
            marker["paint_mode"] = "root"
            marker["paint_root"] = source_roots[0]
        else:
            replacement = ("stripe", source_colors[0], source_colors[1])
            marker["paint_mode"] = "style"
            marker["paint_style"] = replacement
        assert final_style[dest] in {("solid", root_color[dest]), replacement}, "conflicting T controls for one region"
        final_style[dest] = replacement

    out = np.full(g.shape, -1, dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = _style_at_row(final_style[dsu.find(cid)], r)

    for marker in markers:
        if marker["paint_mode"] == "root":
            style = final_style[marker["paint_root"]]
        else:
            style = marker["paint_style"]
        for r, c in marker["cells"]:
            out[r, c] = _style_at_row(style, r)

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
    Resolve T-shaped boundary controls, allowing a split source to create striped recoloring, then optionally rotate via a color-9 L marker.

    Input structure:
    - The grid is partitioned into irregular monochrome regions.
    - Color 1 forms T-shaped controls on region boundaries.
    - A colored center with three singleton color-1 neighbors is still a literal-color T control.
    - A normal T may read either one source side or two different diagonal source regions.
    - An optional 5-cell color-9 L marker is embedded inside one region.

    Rule:
    - The T stem points toward the destination region.
    - For a normal T, inspect the two diagonal cells at `center + stem + each bar direction`.
    - If those diagonals resolve to one source color, repaint the destination solidly with that color as before.
    - If they resolve to two different source colors, repaint the destination with horizontal stripes:
      odd grid rows use the first diagonal source and even grid rows use the second.
    - Colored-center T markers still repaint the destination with the center color.
    - Marker cells are erased into the source-side appearance; for a dual-source normal T, they use the same stripe pattern as the destination.
    - A color-9 L marker still rotates the finished output: ┐ means CCW 90 degrees, ┌ means CW 90 degrees.

    Key insight:
    - The stem still answers "which region changes", but the pair of diagonals now answers
      either "copy one source color" or "blend two visible source colors into row stripes".
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (l9rot + dual-source stripes) ---------------

# ── Tunable constants ──────────────────────────────────────────────────
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_MAX = 0

COLOR_POOL = [c for c in range(10) if c != 1]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 500

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bar(stem):
    """Return the two bar directions perpendicular to *stem* in a stable visible order."""
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center, stem):
    """Return the 4 cells of a T marker: center + stem tip + 2 bar cells."""
    r, c = center
    bar = _get_bar(stem)
    return {
        (r, c),
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_neighborhood(center, stem):
    """Return cells adjacent to any T cell but not part of the T itself."""
    cells = _t_cells(center, stem)
    neighborhood = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _style_at(style: tuple, r: int) -> int:
    if style[0] == "solid":
        return int(style[1])
    return int(style[1] if int(r) % 2 == 0 else style[2])


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    """Grow n_regions from random seeds via randomized BFS; returns region-id array or None."""
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


def _choose_marker_counts(n_regions: int) -> tuple[int, int, int] | None:
    """Return (n_markers, n_normal, n_colored) for the dual-source variant."""
    max_possible = min(MARKER_MAX, n_regions)
    if max_possible < MARKER_MIN:
        return None
    n_markers = random.randint(MARKER_MIN, max_possible)
    n_normal = min(n_markers, NORMAL_MAX)
    return n_normal, n_normal, 0


def _first_destination(region, center, stem, occupied_cells):
    r, c = center
    step = 2
    while True:
        nr, nc = r + stem[0] * step, c + stem[1] * step
        if not (0 <= nr < region.shape[0] and 0 <= nc < region.shape[1]):
            return None
        if (nr, nc) in occupied_cells:
            step += 1
            continue
        return nr, nc, int(region[nr, nc])


def _find_valid_dual_placements(region, occupied_cells, occupied_neighbors, dest_used):
    """Find all dual-source T placements whose diagonals read two different source regions."""
    H, W = region.shape
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in _DIRS:
                bar = _get_bar(stem)
                cells = _t_cells((r, c), stem)
                if cells & occupied_cells:
                    continue
                if _t_neighborhood((r, c), stem) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue

                diag_regions = []
                ok = True
                for bd in bar:
                    rr = r + stem[0] + bd[0]
                    cc = c + stem[1] + bd[1]
                    if not (0 <= rr < H and 0 <= cc < W):
                        ok = False
                        break
                    diag_regions.append(int(region[rr, cc]))
                if not ok or diag_regions[0] == diag_regions[1]:
                    continue

                dest_info = _first_destination(region, (r, c), stem, occupied_cells)
                if dest_info is None:
                    continue
                _, _, dst = dest_info
                if dst in dest_used or dst in diag_regions:
                    continue
                candidates.append(((r, c), stem, diag_regions[0], diag_regions[1], dst))
    return candidates


def _register_marker(center, stem, occupied_cells, occupied_neighbors):
    """Update occupied sets after placing a normal T marker."""
    cells = _t_cells(center, stem)
    occupied_cells |= cells
    occupied_neighbors |= (_t_neighborhood(center, stem) - cells)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _render_region_styles(region: np.ndarray, styles: list[tuple]) -> np.ndarray:
    out = np.empty_like(region)
    for r in range(region.shape[0]):
        for c in range(region.shape[1]):
            out[r, c] = _style_at(styles[int(region[r, c])], r)
    return out


def _stamp_markers_input(grid, placed_markers):
    for _kind, center, stem, _src_a, _src_b, _dst in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = 1


def _stamp_markers_output(grid, placed_markers, out_styles):
    for _kind, center, stem, _src_a, _src_b, dst in placed_markers:
        style = out_styles[dst]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = _style_at(style, cr)


def _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_styles):
    """Attempt to place a 5-cell L9 marker; return orientation string or None."""
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, stem, *_ in placed_markers:
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
        output_grid[lr + dr, lc + dc] = _style_at(out_styles[rid], lr + dr)
    return orient


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    target_markers, _n_normal, _n_colored = counts

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    dest_used: set[int] = set()
    placed_markers = []
    while len(placed_markers) < target_markers:
        cands = _find_valid_dual_placements(region, occupied_cells, occupied_neighbors, dest_used)
        if not cands:
            break
        center, stem, src_a, src_b, dst = random.choice(cands)
        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        dest_used.add(dst)
        placed_markers.append(("normal", center, stem, src_a, src_b, dst))
    if len(placed_markers) < MARKER_MIN:
        return None

    out_styles = [("solid", input_colors[rid]) for rid in range(n_regions)]
    for _kind, _center, _stem, src_a, src_b, dst in placed_markers:
        out_styles[dst] = ("stripe", input_colors[src_a], input_colors[src_b])

    input_grid = _build_color_map(region, input_colors)
    output_grid = _render_region_styles(region, out_styles)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_styles)

    l9_orient = _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_styles)
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
