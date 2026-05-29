# Combo variant of 21897d95 (index 17)
# combo_spec: D6/0+D7/0+D13/1
# Name: 矩形格内嵌目标标记

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the combo variant
# -------------------------------------
# combo_summary:
#   - D6/0 replaces the original irregular connected blobs with a full axis-aligned
#     rectangle tiling obtained from horizontal and vertical split lines.
#   - D7/0 removes the optional L9 marker entirely, so output size and orientation never
#     change.
#   - D13/1 keeps the T inside the destination region, but in this rectangular setting the
#     most legible interpretation is directional: the marker is placed near the side toward
#     which the stem points, and the source region is found by scanning in the opposite
#     direction through the target rectangle until crossing its far side.
#
# concrete_pixel_behavior:
#   - A normal marker is a 4-cell color-1 T tetromino.
#   - A colored marker is the same T shape except that the center keeps a literal non-1
#     replacement color and only the three branch cells are color 1.
#   - Every rectangle receives exactly one marker, so "which region changes" is visually
#     obvious: the marker is embedded inside that rectangle.
#   - For stem RIGHT, the center sits one column left of the rectangle's right edge and the
#     stem tip touches that side. The solver identifies the destination from local adjacency,
#     then scans LEFT until it leaves that rectangle; the first rectangle reached there is
#     the source. LEFT/UP/DOWN are symmetric.
#
# behavior_trace:
#   - Original task, stem RIGHT:
#       destination = first region hit by walking RIGHT from the marker
#       source      = diagonal witness cells on the RIGHT side
#   - This combo, stem RIGHT:
#       destination = rectangle surrounding the marker
#       source      = first rectangle reached by walking LEFT through the destination
#       outward cue = the cell at center + 2*RIGHT is outside the destination
#   - The semantic reversal is therefore larger than plain D13/1: not only is the T inside
#     the target, but the rectangular tessellation turns the reverse direction itself into
#     the source locator.
#
# complexity_calibration:
#   - Region shapes are simpler, so the generator compensates by giving every rectangle a
#     marker and by keeping 4-6 rectangles per grid.
#   - Rectangle side lengths are at least 5 cells. That keeps the reverse-scan distance
#     usually larger than 1, so the combo does not collapse back to the original boundary
#     reading or to the single-variant "immediate back-side diagonal" cue.
#   - Colored-center markers are limited to 0-2 per grid, matching the original task's low
#     cardinality and keeping the control vocabulary small.
#
# hidden_assumption_audit:
#   - `first_component_from(center, stem, start=2)` from the base solver is invalid because
#     the stem no longer points to the destination.
#   - The old diagonal source lookup is also invalid in the fused rectangular rule, because
#     source selection now depends on crossing the destination rectangle in direction `-stem`.
#   - Rotation code and all color-9 assumptions must disappear; otherwise harmless region
#     color 9 values would be misread as orientation controls.
#   - The solver now assumes:
#       1. each T has a unique dominant adjacent region color, which is its destination;
#       2. scanning along `-stem` exits that destination and then reaches exactly one source
#          region color before leaving the grid.
#     The generator enforces both by using full rectangles, one marker per rectangle, and a
#     placement rule that puts the center one cell from the stem-side boundary.
#
# placement_legality:
#   - Markers never overlap and are buffered so color-1 components from different markers do
#     not touch orthogonally across rectangle boundaries.
#   - All T cells remain inside the target rectangle.
#   - The cell beyond the stem tip is outside the target rectangle by construction.
#   - The chosen source side always has an adjacent rectangle, so reverse scanning is valid.
#
# observability_and_cleanliness:
#   - The rectangular mosaic creates stable row/column alignments. Across a few examples a
#     human can see that each marked rectangle changes, while the donor is the rectangle on
#     the opposite side of the stem direction.
#   - No scraps or L markers are present, which keeps the scene visually sparse and avoids
#     overlaying too many object types on the same rectangles.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-int(d[0]), -int(d[1]))


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if stem[0] == 0:
        return [UP, DOWN]
    return [LEFT, RIGHT]


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
    return _t_cells(center, stem) - {center}


def _t_neighborhood(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _t_cells(center, stem)
    neighborhood = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _stem_from_branches(branches: list[tuple[int, int]]) -> tuple[int, int]:
    assert len(branches) == 3
    branch_set = set(branches)
    for d in branches:
        if _opp(d) not in branch_set:
            return d
    raise AssertionError("T marker has no single stem")


def _parse_tetromino(
    cells: set[tuple[int, int]],
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
) -> tuple[np.ndarray, list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
    cid = 0
    for color in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == color) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == k)
            if coords.size == 0:
                continue
            for rr, cc in coords:
                comp_id[int(rr), int(cc)] = cid
            comp_colors.append(int(color))
            cid += 1
    return comp_id, comp_colors


def _adjacent_component_counts(
    comp_id: np.ndarray, cells: set[tuple[int, int]]
) -> Counter:
    counts: Counter = Counter()
    for r, c in cells:
        for dr, dc in DIRS:
            rr, cc = r + dr, c + dc
            if _in_bounds(comp_id.shape, rr, cc) and comp_id[rr, cc] >= 0:
                counts[int(comp_id[rr, cc])] += 1
    return counts


def _scan_source_component(
    comp_id: np.ndarray,
    control: np.ndarray,
    center: tuple[int, int],
    stem: tuple[int, int],
    dest_comp: int,
) -> int | None:
    direction = _opp(stem)
    seen_dest = False
    r0, c0 = center
    step = 1
    while True:
        rr = r0 + direction[0] * step
        cc = c0 + direction[1] * step
        if not _in_bounds(comp_id.shape, rr, cc):
            return None
        if control[rr, cc]:
            step += 1
            continue
        cid = int(comp_id[rr, cc])
        assert cid >= 0, "reverse scan must stay on visible rectangle cells"
        if cid == dest_comp:
            seen_dest = True
            step += 1
            continue
        if not seen_dest:
            return None
        return cid


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
    assert all_one_cells == used_ones, "all color-1 pixels must belong to T markers"

    comp_id, comp_colors = _label_base_components(g, control)
    assert comp_colors, "at least one base rectangle is required"
    final_color = {cid: color for cid, color in enumerate(comp_colors)}
    for marker in markers:
        counts = _adjacent_component_counts(comp_id, marker["cells"])
        assert counts, "embedded T marker must touch its destination rectangle"
        dest = max(counts, key=lambda cid: (counts[cid], cid))
        source = _scan_source_component(comp_id, control, marker["center"], marker["stem"], dest)
        assert source is not None, "reverse stem scan must reach a source rectangle"
        marker["dest_comp"] = int(dest)
        marker["source_comp"] = int(source)
        if marker["kind"] == "colored":
            final_color[dest] = int(marker["target_color"])
        else:
            final_color[dest] = int(comp_colors[source])

    out = np.full(g.shape, -1, dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[cid])

    for marker in markers:
        fill_color = int(final_color[marker["source_comp"]])
        for r, c in marker["cells"]:
            out[r, c] = fill_color

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a rectangular tiling from embedded T controls and erase the controls in place.

    Input structure:
    - The grid is partitioned into 4-6 monochrome rectangles created by horizontal and
      vertical split lines.
    - Every rectangle contains exactly one T-shaped control marker.
    - Normal controls are 4-cell color-1 T tetrominoes; colored controls replace the T's
      center with a literal non-1 color while keeping the three branch cells color 1.
    - There is no color-9 rotation marker in this variant, so input and output keep the same
      size and orientation.

    Rule:
    - The rectangle that locally surrounds a T is the destination rectangle that changes.
    - The stem points outward toward one side of that destination rectangle.
    - To find the source rectangle, walk from the T center in the direction opposite the
      stem until you leave the destination; the first rectangle reached there is the donor.
    - A normal T copies the donor rectangle's original color into the destination; a
      colored-center T uses its center color instead.
    - All four cells of the T are overwritten by the donor rectangle's final output color.

    Key insight:
    - In this fusion the T no longer sits on a boundary and it does not aim at the thing
      that changes. The marked rectangle itself is what changes, while the rectangle on the
      far side of the reverse stem direction supplies the replacement color.
    """
    g = np.asarray(grid, dtype=int)
    return _solve_unrotated(g)


# --------------- generator (rect_combo) ---------------

GRID_SIZE_MAX = 30
MIN_RECT_SIDE = 5
COLORED_MAX = 2
GEN_RETRIES = 400
COLOR_POOL = [c for c in range(10) if c != 1]

LAYOUTS = [
    (2, 2),
    (2, 3),
    (3, 2),
]
LAYOUT_WEIGHTS = [4, 3, 3]


def _random_segments(total: int, parts: int, min_size: int) -> list[int] | None:
    if total < parts * min_size:
        return None
    sizes = [min_size] * parts
    for _ in range(total - parts * min_size):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _build_rect_partition(
    H: int, W: int, row_parts: int, col_parts: int
) -> tuple[np.ndarray, list[dict]]:
    row_sizes = _random_segments(H, row_parts, MIN_RECT_SIDE)
    col_sizes = _random_segments(W, col_parts, MIN_RECT_SIDE)
    assert row_sizes is not None and col_sizes is not None

    row_edges = [0]
    for h in row_sizes:
        row_edges.append(row_edges[-1] + h)
    col_edges = [0]
    for w in col_sizes:
        col_edges.append(col_edges[-1] + w)

    region = np.full((H, W), -1, dtype=int)
    rects: list[dict] = []
    rid = 0
    for rr in range(row_parts):
        for cc in range(col_parts):
            r0, r1 = row_edges[rr], row_edges[rr + 1]
            c0, c1 = col_edges[cc], col_edges[cc + 1]
            region[r0:r1, c0:c1] = rid
            rects.append({"rid": rid, "row": rr, "col": cc, "bbox": (r0, r1, c0, c1)})
            rid += 1
    return region, rects


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    return np.asarray(colors, dtype=int)[region]


def _candidate_sources(
    rect: dict, row_parts: int, col_parts: int
) -> list[tuple[int, tuple[int, int]]]:
    rr = int(rect["row"])
    cc = int(rect["col"])
    candidates: list[tuple[int, tuple[int, int]]] = []
    if cc > 0:
        candidates.append((rr * col_parts + (cc - 1), RIGHT))
    if cc + 1 < col_parts:
        candidates.append((rr * col_parts + (cc + 1), LEFT))
    if rr > 0:
        candidates.append(((rr - 1) * col_parts + cc, DOWN))
    if rr + 1 < row_parts:
        candidates.append(((rr + 1) * col_parts + cc, UP))
    random.shuffle(candidates)
    return candidates


def _rect_centers_for_stem(rect: dict, stem: tuple[int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = rect["bbox"]
    centers: list[tuple[int, int]] = []
    if stem == RIGHT:
        col = c1 - 2
        for row in range(r0 + 2, r1 - 2):
            centers.append((row, col))
    elif stem == LEFT:
        col = c0 + 1
        for row in range(r0 + 2, r1 - 2):
            centers.append((row, col))
    elif stem == DOWN:
        row = r1 - 2
        for col in range(c0 + 2, c1 - 2):
            centers.append((row, col))
    elif stem == UP:
        row = r0 + 1
        for col in range(c0 + 2, c1 - 2):
            centers.append((row, col))
    return centers


def _find_rect_marker_candidates(
    rect: dict,
    row_parts: int,
    col_parts: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for src, stem in _candidate_sources(rect, row_parts, col_parts):
        for center in _rect_centers_for_stem(rect, stem):
            cells = _t_cells(center, stem)
            if cells & occupied_cells:
                continue
            if _t_neighborhood(center, stem) & occupied_cells:
                continue
            if cells & occupied_neighbors:
                continue
            candidates.append((src, center, stem))
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


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[tuple]) -> None:
    for kind, center, stem, _src, _dst, target_color in placed_markers:
        if kind == "normal":
            for r, c in _t_cells(center, stem):
                grid[r, c] = 1
        else:
            grid[center[0], center[1]] = int(target_color)
            for r, c in _t_branch_cells(center, stem):
                grid[r, c] = 1


def _stamp_markers_output(
    grid: np.ndarray, placed_markers: list[tuple], out_colors: list[int]
) -> None:
    for _kind, center, stem, src, _dst, _target_color in placed_markers:
        fill = int(out_colors[src])
        for r, c in _t_cells(center, stem):
            grid[r, c] = fill


def _has_balanced_branch_coverage(placed_markers: list[tuple]) -> bool:
    stems = {stem for _kind, _center, stem, _src, _dst, _target_color in placed_markers}
    has_horizontal = LEFT in stems or RIGHT in stems
    has_vertical = UP in stems or DOWN in stems
    if not (has_horizontal and has_vertical):
        return False
    if len(placed_markers) >= 6 and len(stems) < 3:
        return False
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    row_parts, col_parts = random.choices(LAYOUTS, weights=LAYOUT_WEIGHTS)[0]
    n_regions = row_parts * col_parts

    min_h = row_parts * MIN_RECT_SIDE
    min_w = col_parts * MIN_RECT_SIDE
    if min_h > GRID_SIZE_MAX or min_w > GRID_SIZE_MAX:
        return None

    H = random.randint(max(9, min_h), GRID_SIZE_MAX)
    W = random.randint(max(9, min_w), GRID_SIZE_MAX)
    region, rects = _build_rect_partition(H, W, row_parts, col_parts)

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    n_colored = random.randint(0, min(COLORED_MAX, n_regions - 2))
    colored_rids = set(random.sample(range(n_regions), n_colored)) if n_colored else set()
    marker_specs = []
    for rid in range(n_regions):
        kind = "colored" if rid in colored_rids else "normal"
        target_color = None
        if kind == "colored":
            color_choices = [c for c in pool if c != input_colors[rid]]
            if not color_choices:
                return None
            target_color = random.choice(color_choices)
        marker_specs.append((rid, kind, target_color))
    random.shuffle(marker_specs)

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers: list[tuple] = []
    pending = marker_specs[:]
    while pending:
        best_idx = -1
        best_candidates: list[tuple[int, tuple[int, int], tuple[int, int]]] | None = None
        for idx, (rid, kind, target_color) in enumerate(pending):
            cands = _find_rect_marker_candidates(
                rects[rid], row_parts, col_parts, occupied_cells, occupied_neighbors
            )
            if not cands:
                return None
            if best_candidates is None or len(cands) < len(best_candidates):
                best_idx = idx
                best_candidates = cands
        rid, kind, target_color = pending.pop(best_idx)
        assert best_candidates is not None
        src, center, stem = random.choice(best_candidates)
        _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, rid, target_color))

    if not _has_balanced_branch_coverage(placed_markers):
        return None

    out_colors = list(input_colors)
    reverse_scan_distances: list[int] = []
    for kind, center, stem, src, dst, target_color in placed_markers:
        if kind == "colored":
            out_colors[dst] = int(target_color)
        else:
            out_colors[dst] = int(input_colors[src])
        rect = rects[dst]["bbox"]
        r0, r1, c0, c1 = rect
        if stem == RIGHT:
            reverse_scan_distances.append(center[1] - c0 + 1)
        elif stem == LEFT:
            reverse_scan_distances.append(c1 - center[1])
        elif stem == DOWN:
            reverse_scan_distances.append(center[0] - r0 + 1)
        else:
            reverse_scan_distances.append(r1 - center[0])

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    meta = {
        "n_regions": n_regions,
        "n_colored": n_colored,
        "reverse_scan_distances": reverse_scan_distances,
        "layout": (row_parts, col_parts),
    }
    return input_grid, output_grid, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        generated = _gen_attempt()
        if generated is None:
            continue
        input_grid, output_grid, _meta = generated
        try:
            result = main(input_grid)
        except Exception:
            continue
        if np.array_equal(result, output_grid):
            return input_grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
