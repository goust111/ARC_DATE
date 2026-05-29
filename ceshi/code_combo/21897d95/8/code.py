# Combo variant of 21897d95 (index 8)
# combo_spec: D6/2+D13/0
# Name: 条纹内嵌标记
#
# Design analysis for the combo variant
# - Concrete behavior:
#   The irregular mosaic is replaced by 4-5 full-width or full-height stripes whose
#   thicknesses are intentionally unequal. A T marker no longer straddles a border: all
#   four cells of the T sit inside one source stripe, and the stem points toward the nearest
#   stripe boundary. Following that ray through the stripe interior reaches the adjacent
#   destination stripe on the stem side.
# - Behavioral trace:
#   Original task: a boundary T could identify the source from the two diagonals at
#   `center + stem + bar`, and the destination appeared almost immediately after the stem.
#   This combo: the same visible T shape is suspended inside one stripe. The source must be
#   inferred from the surrounding stripe color, and the destination is the first foreign
#   stripe reached only after traversing interior source cells in the stem direction. Because
#   stripes are linear, only up/down stems are valid in horizontal layouts and only left/right
#   stems are valid in vertical layouts.
# - Complexity calibration:
#   The generator enforces `stem_exit >= 3`, so every marker has at least one full source cell
#   between the stem tip and the boundary. It also enforces that the stem side is the unique
#   nearest exit direction, which makes the internal placement visually meaningful instead of a
#   disguised boundary marker. Stripe widths vary so the nearest-side cue is observable across
#   multiple examples.
# - Implicit assumptions audited from the original solver:
#   `first_component_from(..., start=2)` is no longer valid because the destination is not
#   adjacent to the T. Source lookup from boundary diagonals is also invalid because the T
#   never touches the destination boundary. The old "small scrap" heuristic is removed: an
#   internal T or L9 hole can split one stripe into tiny same-color fragments, and those must
#   be merged back into the owner stripe rather than misclassified as standalone scraps.
# - Placement legality:
#   Generator checks guarantee that the whole T and its 4-neighborhood remain inside one
#   stripe, markers do not overlap or crowd each other, the corridor from the stem to the
#   boundary is clear, and the first foreign stripe on that corridor is exactly the intended
#   destination. L9, when present, is also placed fully inside one stripe with a buffer.
# - Visual observability:
#   A human sees repeated full-width/full-height bands plus T shapes floating inside a band.
#   The recurring pattern is that the repaint target is the adjacent stripe on the closest
#   stem side, not a region touching the T itself. Unequal stripe widths make the "nearest
#   boundary" cue visible in only one axis, which is easier to infer from 2-4 examples.
# - Visual cleanliness:
#   Region colors are distinct in the input and color 9 is reserved for the optional L marker,
#   so the stripe structure, T controls, and L9 rotation cue remain separable. Buffer cells
#   prevent stacked markers from cluttering a single stripe corridor.
# - Color-change inferability:
#   Normal T markers copy a neighboring stripe's original color; colored-center T markers use
#   a fresh literal center color that is not reused by any stripe in the input. This keeps the
#   recoloring rule structural rather than dependent on arbitrary color identities.
# - New solver assumptions and generator guarantees:
#   The solver assumes every T or L9 control object is fully embedded in exactly one stripe
#   color, that a T stem has a unique nearest foreign exit, and that color 9 belongs only to
#   the optional L marker. The generator constructs inputs to satisfy each of those conditions.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
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


def _find_l9_marker(g: np.ndarray) -> dict | None:
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    found: dict | None = None
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
            if h_nb not in coords or v_nb not in coords:
                continue
            if cr == r0 and cc == c1:
                orient = "top_right"
            elif cr == r0 and cc == c0:
                orient = "top_left"
            else:
                continue
            assert found is None, "at most one L9 marker is allowed"
            found = {"kind": "l9", "orientation": orient, "cells": coords}
    return found


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
            {"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells}
        )
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) in {1, 9}:
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

    l9_marker = _find_l9_marker(g)
    if l9_marker is not None:
        markers.append(l9_marker)
        for r, c in l9_marker["cells"]:
            control[r, c] = True

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
        assert counts, "embedded control object must touch its owner stripe"
        owner = max(counts, key=lambda cid: (counts[cid], comp_areas[cid]))
        owner_color = int(comp_colors[owner])
        assert all(
            int(comp_colors[cid]) == owner_color for cid in counts
        ), "embedded control object must be surrounded by one stripe color"
        for cid in counts:
            dsu.union(owner, cid)
        marker["source_comp"] = owner

    for marker in markers:
        attach_owner(marker)

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged stripe fragments must share a color"

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
            assert seen_source, "stem must leave its own stripe before reaching a destination"
            return cid, step

    final_color = dict(root_color)
    for marker in markers:
        source_root = dsu.find(marker["source_comp"])
        marker["source"] = source_root
        if marker["kind"] not in {"normal", "colored"}:
            continue

        exit_distances: dict[tuple[int, int], int] = {}
        for direction in DIRS:
            result = first_foreign_component_from(marker["center"], direction, source_root)
            if result is not None:
                _cid, steps = result
                exit_distances[direction] = int(steps)

        stem = marker["stem"]
        assert stem in exit_distances, "T stem must eventually reach a destination stripe"
        stem_dist = exit_distances[stem]
        assert all(
            exit_distances[d] > stem_dist for d in exit_distances if d != stem
        ), "T stem must indicate the unique nearest exit direction"

        dest_result = first_foreign_component_from(marker["center"], stem, source_root)
        assert dest_result is not None
        dest_comp, _ = dest_result
        dest = dsu.find(dest_comp)
        replacement = (
            int(marker["target_color"])
            if marker["kind"] == "colored"
            else int(root_color[source_root])
        )
        assert final_color[dest] in {
            root_color[dest],
            replacement,
        }, "conflicting T controls for one destination stripe"
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
    marker = _find_l9_marker(g)
    return None if marker is None else str(marker["orientation"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint stripe bands from T-shaped controls that float inside a source stripe, then optionally rotate via a color-9 L marker.

    Input structure:
    - The grid is partitioned into unequal horizontal or vertical monochrome stripes.
    - Color 1 forms T-shaped controls, and some controls use a colored center with three singleton color-1 branches.
    - Every T lies fully inside one stripe instead of on a stripe boundary.
    - An optional 5-cell L-shaped color-9 marker is also fully embedded inside one stripe.

    Rule:
    - The stripe surrounding a T is the source stripe.
    - The T stem points toward the nearest stripe boundary; following that ray through the source interior until the first different stripe identifies the destination stripe.
    - A normal T repaints the destination to the source stripe's original color, while a colored-center T repaints it to the center color.
    - All T cells and all L9 cells are filled with the final color of the stripe that contains them.
    - A 5-cell L-shaped color-9 marker controls rotation: top-right corner means rotate 90 degrees counterclockwise, and top-left corner means rotate 90 degrees clockwise.

    Key insight:
    - The T no longer marks a boundary crossing directly; instead, you must read the stripe geometry and use the stem to choose the adjacent stripe reached at the closest edge of the T's own stripe.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (stripe-internal-T + l9rot) ---------------

GRID_SIZE_MAX = 30
MIN_CROSS_AXIS = 7
MIN_STRIPE_THICK = 6
STRIPE_COUNTS = [4, 5]
STRIPE_WEIGHTS = [2, 3]

MARKER_MIN = 4
MARKER_MAX = 5
NORMAL_MAX = 5
COLORED_MAX = 2

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

GEN_RETRIES = 400
MIN_STEM_EXIT = 3


def _shape_neighborhood(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    neighborhood = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _random_composition(total: int, parts: int, min_part: int) -> list[int] | None:
    extra = total - parts * min_part
    if extra < 0:
        return None
    cuts = sorted(random.sample(range(extra + parts - 1), parts - 1)) if parts > 1 else []
    widths: list[int] = []
    prev = -1
    for cut in cuts + [extra + parts - 1]:
        widths.append(min_part + cut - prev - 1)
        prev = cut
    return widths


def _make_stripe_partition(
    orientation: str, H: int, W: int, widths: list[int]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    region = np.full((H, W), -1, dtype=int)
    spans: list[tuple[int, int]] = []
    pos = 0
    for rid, width in enumerate(widths):
        start, end = pos, pos + width - 1
        spans.append((start, end))
        if orientation == "horizontal":
            region[start : end + 1, :] = rid
        else:
            region[:, start : end + 1] = rid
        pos = end + 1
    return region, spans


def _build_stripes() -> tuple[np.ndarray, str, list[tuple[int, int]]] | None:
    orientation = random.choice(["horizontal", "vertical"])
    n_regions = random.choices(STRIPE_COUNTS, weights=STRIPE_WEIGHTS)[0]
    axis_min = n_regions * MIN_STRIPE_THICK
    if axis_min > GRID_SIZE_MAX:
        return None
    axis = random.randint(axis_min, GRID_SIZE_MAX)
    cross = random.randint(MIN_CROSS_AXIS, GRID_SIZE_MAX)
    widths = _random_composition(axis, n_regions, MIN_STRIPE_THICK)
    if widths is None or len(set(widths)) < 2:
        return None
    if orientation == "horizontal":
        region, spans = _make_stripe_partition(orientation, axis, cross, widths)
    else:
        region, spans = _make_stripe_partition(orientation, cross, axis, widths)
    return region, orientation, spans


def _choose_marker_counts(n_regions: int) -> tuple[int, int, int]:
    n_markers = random.randint(MARKER_MIN, min(MARKER_MAX, n_regions))
    n_colored = random.randint(0, min(COLORED_MAX, n_markers - 2))
    n_normal = n_markers - n_colored
    if n_normal > NORMAL_MAX:
        n_normal = NORMAL_MAX
        n_colored = n_markers - n_normal
    return n_markers, n_normal, n_colored


def _neighbor_sources(dst: int, n_regions: int) -> list[int]:
    sources = []
    if dst - 1 >= 0:
        sources.append(dst - 1)
    if dst + 1 < n_regions:
        sources.append(dst + 1)
    return sources


def _choose_marker_pairs(n_regions: int, n_markers: int) -> list[tuple[int, int]] | None:
    destinations = random.sample(range(n_regions), n_markers)
    src_use = [0] * n_regions
    pairs: list[tuple[int, int]] = []
    for dst in destinations:
        cands = _neighbor_sources(dst, n_regions)
        random.shuffle(cands)
        cands.sort(key=lambda src: (src_use[src], random.random()))
        if not cands:
            return None
        src = cands[0]
        src_use[src] += 1
        pairs.append((src, dst))
    return pairs


def _direction_for_pair(orientation: str, src: int, dst: int) -> tuple[int, int]:
    if orientation == "horizontal":
        return UP if dst < src else DOWN
    return LEFT if dst < src else RIGHT


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
    orientation: str,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    H, W = region.shape
    stem = _direction_for_pair(orientation, src, dst)
    candidates = []
    for r in range(H):
        for c in range(W):
            if int(region[r, c]) != src:
                continue
            center = (r, c)
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
            if cells & occupied_neighbors:
                continue
            if neighborhood & occupied_cells:
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

            if not _corridor_clear(
                center, stem, stem_dist, cells, occupied_cells, occupied_neighbors
            ):
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
    return np.asarray(colors, dtype=int)[region]


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[tuple]) -> None:
    for kind, center, stem, _src, _dst, target_color in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = int(target_color)
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(
    grid: np.ndarray, placed_markers: list[tuple], out_colors: list[int]
) -> None:
    for _kind, center, stem, src, _dst, _target_color in placed_markers:
        color = int(out_colors[src])
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> str | None:
    if random.random() >= L9_PROBABILITY:
        return None
    H, W = region.shape
    orientations = list(L9_OFFSETS.items())
    random.shuffle(orientations)
    for orient, offsets in orientations:
        candidates = []
        for r in range(H):
            for c in range(W):
                cells = {(r + dr, c + dc) for dr, dc in offsets}
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                rid = int(region[r, c])
                if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                    continue
                neighborhood = _shape_neighborhood(cells)
                if not all(
                    0 <= nr < H and 0 <= nc < W and int(region[nr, nc]) == rid
                    for nr, nc in neighborhood
                ):
                    continue
                if cells & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue
                if neighborhood & occupied_cells:
                    continue
                candidates.append((cells, rid))
        if not candidates:
            continue
        cells, rid = random.choice(candidates)
        for cr, cc in cells:
            input_grid[cr, cc] = 9
            output_grid[cr, cc] = int(out_colors[rid])
        return orient
    return None


def _gen_attempt() -> np.ndarray | None:
    built = _build_stripes()
    if built is None:
        return None
    region, orientation, spans = built
    n_regions = len(spans)

    n_markers, n_normal, n_colored = _choose_marker_counts(n_regions)
    marker_pairs = _choose_marker_pairs(n_regions, n_markers)
    if marker_pairs is None:
        return None

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    input_colors = pool[:n_regions]
    fresh_colors = [c for c in pool if c not in input_colors]
    if len(fresh_colors) < n_colored:
        return None

    out_colors = list(input_colors)
    normal_pairs = marker_pairs[:n_normal]
    colored_pairs = marker_pairs[n_normal:]
    colored_targets = random.sample(fresh_colors, n_colored)

    for src, dst in normal_pairs:
        out_colors[dst] = input_colors[src]
    for (_src, dst), target in zip(colored_pairs, colored_targets):
        out_colors[dst] = int(target)

    all_defs = [("normal", s, d, None) for s, d in normal_pairs] + [
        ("colored", s, d, target)
        for (s, d), target in zip(colored_pairs, colored_targets)
    ]
    random.shuffle(all_defs)

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers = []
    for kind, src, dst, target in all_defs:
        candidates = _find_valid_placements(
            region, orientation, src, dst, occupied_cells, occupied_neighbors
        )
        if not candidates:
            return None
        center, stem = random.choice(candidates)
        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, target))

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)
    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    l9_orient = _try_place_l9(
        region,
        input_grid,
        output_grid,
        out_colors,
        occupied_cells,
        occupied_neighbors,
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
