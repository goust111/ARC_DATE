# Combo variant of 21897d95 (index 15)
# combo_spec: D1/1+D11/2
# Name: 跨调色板混色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis for the fused variant
# =====================================
# a) Concrete behavior:
#    Original task: a T marker copies one region color into another destination region,
#    or a colored-center T uses its center color as the replacement.
#    New task: the regions themselves still determine source and destination, but a fixed
#    2x3 key in the top-left corner now lists the ordered input palette A on the first
#    row and the ordered output palette B on the second row. A destination no longer
#    receives the source/replacement color directly. Instead, the operative input rank
#    (source-region rank for a normal T, center-color rank for a colored-center T) and
#    the target rank in A are averaged, and the result is read from B. Example:
#      key top    = [orange, green, yellow]   -> ranks 0,1,2
#      key bottom = [red,    gray,  blue]     -> ranks 0,1,2
#      if a T uses a rank-0 source and points at a rank-2 destination, the output rank
#      is 1, so the destination becomes gray rather than orange or yellow.
#    The visible difference from the original task is that the destination color can be
#    a third palette-B color that is not present in the destination or the source region
#    in the input mosaic.
#
# b) Complexity calibration:
#    The average rule is only used on exact-middle cases. The generator therefore uses a
#    small 3-color palette and only places normal markers on same-parity rank pairs; for
#    colored-center markers it chooses a center rank with the same parity as the target.
#    This avoids hidden numeric rounding while still making the recoloring visibly
#    different from plain copying.
#
# c) Broken assumptions from the original solver:
#    - Original final colors were existing region colors or a colored center. That no
#      longer holds: final colors now come from palette B via rank lookup.
#    - Original solver could infer everything from T geometry alone. Here it also needs
#      the explicit palette key because a single test input must reveal the A<->B mapping.
#    - Original generator had both normal and colored-center T controls. The fused
#      version still needs both branches, only with mixed-rank recoloring instead of
#      literal copying/replacement.
#
# d) Placement legality:
#    - The palette key occupies a fixed 2x3 block and is excluded from region growth.
#    - T markers avoid the key, other markers, and each other's immediate neighborhoods.
#    - The optional color-9 L marker also avoids the key and T markers and stays inside
#      a single region.
#
# e) Visual observability:
#    The ordered key makes the cross-palette correspondence visible inside every single
#    puzzle instance, and the T markers still visibly indicate source/destination roles.
#    The nontrivial rank-0 <-> rank-2 cases visibly land on the middle B color, which
#    distinguishes the task from a plain copy rule for both T variants.
#
# f) Visual cleanliness:
#    The grid only contains four object types: large monochrome regions, normal or
#    colored-center T markers, an optional color-9 L marker, and one compact palette key
#    in the corner. The generator keeps these objects separated so the scene stays readable.
#
# g) New solver assumptions and generator guarantees:
#    - The top-left 2x3 block is always the palette key.
#    - Every region color belongs to the top key row.
#    - Every normal T uses a source/destination pair whose ranks have the same parity, and
#      every colored-center T uses a center rank with the same parity as its destination.
#    - Every generated grid contains both normal and colored-center T markers.
#    - At least one normal T and one colored-center T per grid perform a nontrivial 0/2
#      mix, so the fused behavior does not collapse to the original copy rule.


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

PALETTE_SIZE = 3
KEY_ROWS = 2
KEY_COLS = PALETTE_SIZE
KEY_TOP_LEFT = (0, 0)

GRID_SIZE_MAX = 28
GRID_SIZE_MIN = 12
MIN_REGION_CELLS = 8
REGION_COUNT = 7
REGION_SIZE_FACTOR = 2.2
NORMAL_MARKER_COUNT = 2
COLORED_MARKER_COUNT = 2
MARKER_COUNT = NORMAL_MARKER_COUNT + COLORED_MARKER_COUNT
L9_PROBABILITY = 0.5
GEN_RETRIES = 400
SEED_ATTEMPTS = 300

COLOR_POOL = [2, 3, 4, 5, 6, 7, 8]
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}


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


def _palette_key_cells() -> set[tuple[int, int]]:
    r0, c0 = KEY_TOP_LEFT
    return {
        (r0 + dr, c0 + dc)
        for dr in range(KEY_ROWS)
        for dc in range(KEY_COLS)
    }


def _palette_key_buffer(shape: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = KEY_TOP_LEFT
    out = set()
    for r in range(max(0, r0), min(shape[0], r0 + KEY_ROWS + 2)):
        for c in range(max(0, c0), min(shape[1], c0 + KEY_COLS + 2)):
            out.add((r, c))
    return out


def _read_palette_key(
    g: np.ndarray,
) -> tuple[list[int], list[int], dict[int, int], dict[int, int], set[tuple[int, int]]]:
    r0, c0 = KEY_TOP_LEFT
    top = [int(x) for x in g[r0, c0 : c0 + KEY_COLS]]
    bottom = [int(x) for x in g[r0 + 1, c0 : c0 + KEY_COLS]]
    assert len(set(top)) == PALETTE_SIZE, "input palette key row must have distinct colors"
    assert len(set(bottom)) == PALETTE_SIZE, "output palette key row must have distinct colors"
    assert set(top).isdisjoint(bottom), "input/output palettes must be disjoint"
    assert all(col not in {1, 9} for col in top + bottom), "palette key cannot use marker colors"
    input_to_rank = {col: idx for idx, col in enumerate(top)}
    output_to_rank = {col: idx for idx, col in enumerate(bottom)}
    return top, bottom, input_to_rank, output_to_rank, _palette_key_cells()


def _label_base_components(
    g: np.ndarray, control: np.ndarray, input_palette: list[int]
) -> tuple[np.ndarray, list[int], list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
    comp_areas: list[int] = []
    cid = 0
    for col in input_palette:
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


def _find_l9_marker(g: np.ndarray) -> tuple[str | None, set[tuple[int, int]]]:
    mask9 = g == 9
    if not mask9.any():
        return None, set()
    labels, n = label_mask(mask9, connectivity=4)
    found: tuple[str | None, set[tuple[int, int]]] = (None, set())
    for k in range(1, int(n) + 1):
        coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == k)}
        if len(coords) != 5:
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        for cr, cc in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
            h_nb = (cr, cc + 1 if cc == c0 else cc - 1)
            v_nb = (cr + 1 if cr == r0 else cr - 1, cc)
            if h_nb in coords and v_nb in coords:
                if cr == r0 and cc == c1:
                    orient = "top_right"
                elif cr == r0 and cc == c0:
                    orient = "top_left"
                else:
                    continue
                assert found[0] is None, "at most one L9 marker is allowed"
                found = (orient, coords)
    return found


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
    input_palette, output_palette, input_to_rank, _, key_cells = _read_palette_key(g)
    control = np.zeros(g.shape, dtype=bool)
    for r, c in key_cells:
        control[r, c] = True

    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    one_components = {
        k: {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == k)}
        for k in range(1, int(n_1) + 1)
    }
    one_sizes = {k: len(cells) for k, cells in one_components.items()}

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
            if int(g[r, c]) in {1, 9} or (r, c) in key_cells:
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
                color = int(g[r, c])
                assert color in input_to_rank, "colored-center T must use palette-A colors"
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
                        "mix_rank": int(input_to_rank[color]),
                    }
                )
                used_ones |= set(cells)
                for rr, cc in marker_cells:
                    control[rr, cc] = True
    assert used_ones == {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}

    l9_orient, l9_cells = _find_l9_marker(g)
    for r, c in l9_cells:
        control[r, c] = True

    comp_id, comp_colors, comp_areas = _label_base_components(g, control, input_palette)
    assert comp_colors, "at least one base region is required"
    assert all(col in input_to_rank for col in comp_colors), "all region colors must belong to palette A"

    dsu = _DSU(len(comp_colors))

    def first_component_from(
        center: tuple[int, int], direction: tuple[int, int], start: int
    ) -> int | None:
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
            assert comp_colors[part] == comp_colors[source], "T source diagonals must share one input color"
            dsu.union(source, part)

        for comp in adjacent_component_counts(marker["cells"]):
            if comp_colors[comp] == comp_colors[source]:
                dsu.union(source, comp)
        marker["dest_comp"] = dest
        marker["source_comp"] = source

    l9_owner_comp: int | None = None
    if l9_cells:
        counts = adjacent_component_counts(l9_cells)
        assert counts, "L9 marker must touch an owner region"
        owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
        for comp in counts:
            if comp_colors[comp] == comp_colors[owner]:
                dsu.union(owner, comp)
        l9_owner_comp = owner

    root_input_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_input_color.setdefault(root, int(col))
        assert root_input_color[root] == int(col), "merged regions must share one input color"

    root_rank = {root: input_to_rank[col] for root, col in root_input_color.items()}
    final_rank = dict(root_rank)

    for marker in markers:
        dest = dsu.find(marker["dest_comp"])
        source = dsu.find(marker["source_comp"])
        marker["source_root"] = source
        input_rank = int(root_rank[source]) if marker["kind"] == "normal" else int(marker["mix_rank"])
        mix_sum = input_rank + int(root_rank[dest])
        assert mix_sum % 2 == 0, "generator must only use exact midpoint palette pairs"
        mixed_rank = mix_sum // 2
        assert final_rank[dest] in {root_rank[dest], mixed_rank}, "conflicting T controls for one region"
        final_rank[dest] = mixed_rank

    out = np.full(g.shape, -1, dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(output_palette[final_rank[dsu.find(cid)]])

    for marker in markers:
        color = int(output_palette[final_rank[marker["source_root"]]])
        for r, c in marker["cells"]:
            out[r, c] = color

    if l9_owner_comp is not None:
        owner_root = dsu.find(l9_owner_comp)
        color = int(output_palette[final_rank[owner_root]])
        for r, c in l9_cells:
            out[r, c] = color

    r0, c0 = KEY_TOP_LEFT
    out[r0, c0 : c0 + KEY_COLS] = np.array(input_palette, dtype=int)
    out[r0 + 1, c0 : c0 + KEY_COLS] = np.array(output_palette, dtype=int)

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor boundary-marked regions by averaging palette positions, then optionally rotate with an L-shaped 9 marker.

    Input structure:
    - The main canvas is a mosaic of monochrome regions colored only from palette A.
    - A fixed 2x3 key in the top-left corner lists palette A in order on the first row and the corresponding output palette B on the second row.
    - Color 1 forms either 4-cell normal T markers or three singleton branches around a colored-center T, and an optional 5-cell L of color 9 may sit inside one region.

    Rule:
    - For each T marker, the stem points toward the destination region.
    - The two diagonal cells behind the stem identify the source-side region.
    - For a normal T, read the source-region color and destination color as column indices in the top key row.
    - For a colored-center T, read the center color and destination color as column indices in the top key row.
    - Average those two indices and recolor the destination with the color in the same column of the bottom key row.
    - Every ordinary region is shown in palette B in the output, T-marker cells are filled with the final color of their source-side region, and a top-right/top-left L9 rotates the finished output 90 degrees CCW/CW.

    Key insight:
    - The T geometry still tells you which two regions interact, but the corner key tells you how to translate input colors into ordered ranks and then back into the output palette after mixing.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9_orient, _ = _find_l9_marker(g)
    if l9_orient == "top_right":
        return np.rot90(out, 1)
    if l9_orient == "top_left":
        return np.rot90(out, 3)
    return out


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
    out = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                out.add(nb)
    return out


def _grow_regions(H: int, W: int, n_regions: int, blocked: set[tuple[int, int]]) -> np.ndarray | None:
    region = np.full((H, W), -1, dtype=int)
    free_cells = [(r, c) for r in range(H) for c in range(W) if (r, c) not in blocked]
    if len(free_cells) < n_regions * MIN_REGION_CELLS:
        return None

    seed_pool = [(r, c) for r, c in free_cells if 1 <= r < H - 1 and 1 <= c < W - 1]
    random.shuffle(seed_pool)
    seeds = seed_pool[:n_regions]
    if len(seeds) < n_regions:
        return None

    frontier: list[tuple[int, int, int]] = []
    for rid, (r, c) in enumerate(seeds):
        region[r, c] = rid
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(region.shape, nr, nc) and region[nr, nc] == -1 and (nr, nc) not in blocked:
                frontier.append((nr, nc, rid))
    random.shuffle(frontier)

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if region[r, c] != -1 or (r, c) in blocked:
            continue
        region[r, c] = rid
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if _in_bounds(region.shape, nr, nc) and region[nr, nc] == -1 and (nr, nc) not in blocked:
                frontier.append((nr, nc, rid))

    if any(region[r, c] < 0 for r, c in free_cells):
        return None
    for rid in range(n_regions):
        mask = region == rid
        if int(mask.sum()) < MIN_REGION_CELLS:
            return None
        _, n = label_mask(mask, connectivity=4)
        if n != 1:
            return None
    return region


def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    H, W = region.shape
    out = set()
    for r in range(H):
        for c in range(W):
            a = int(region[r, c])
            if a < 0:
                continue
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if not _in_bounds(region.shape, nr, nc):
                    continue
                b = int(region[nr, nc])
                if b >= 0 and a != b:
                    out.add((a, b))
    return list(out)


def _choose_region_ranks(n_regions: int) -> list[int]:
    base = [0, 0, 1, 1, 2, 2]
    extra = [1] * max(0, n_regions - len(base))
    ranks = base + extra
    assert len(ranks) == n_regions
    random.shuffle(ranks)
    return ranks


def _choose_marker_defs(
    directed_pairs: list[tuple[int, int]], region_ranks: list[int]
) -> list[tuple[str, int, int, int | None]] | None:
    normal_candidates = [
        (src, dst)
        for src, dst in directed_pairs
        if src != dst and (region_ranks[src] + region_ranks[dst]) % 2 == 0
    ]
    colored_candidates = [(src, dst) for src, dst in directed_pairs if src != dst]
    if len(normal_candidates) < NORMAL_MARKER_COUNT or len(colored_candidates) < COLORED_MARKER_COUNT:
        return None

    for _ in range(300):
        random.shuffle(normal_candidates)
        random.shuffle(colored_candidates)
        chosen_normals: list[tuple[int, int]] = []
        chosen_colored: list[tuple[int, int, int]] = []
        used_dst: set[int] = set()
        saw_normal_nontrivial = False
        saw_colored_nontrivial = False

        for src, dst in normal_candidates:
            if dst in used_dst:
                continue
            chosen_normals.append((src, dst))
            used_dst.add(dst)
            if abs(region_ranks[src] - region_ranks[dst]) == 2:
                saw_normal_nontrivial = True
            if len(chosen_normals) == NORMAL_MARKER_COUNT:
                break
        if len(chosen_normals) != NORMAL_MARKER_COUNT:
            continue

        for src, dst in colored_candidates:
            if dst in used_dst:
                continue
            dst_rank = int(region_ranks[dst])
            mix_rank = 2 - dst_rank if dst_rank in {0, 2} else 1
            chosen_colored.append((src, dst, mix_rank))
            used_dst.add(dst)
            if abs(mix_rank - dst_rank) == 2:
                saw_colored_nontrivial = True
            if len(chosen_colored) == COLORED_MARKER_COUNT:
                break

        if (
            len(chosen_colored) == COLORED_MARKER_COUNT
            and saw_normal_nontrivial
            and saw_colored_nontrivial
        ):
            defs: list[tuple[str, int, int, int | None]] = [
                ("normal", src, dst, None) for src, dst in chosen_normals
            ]
            defs.extend(
                ("colored", src, dst, mix_rank) for src, dst, mix_rank in chosen_colored
            )
            random.shuffle(defs)
            return defs
    return None


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    blocked_cells: set[tuple[int, int]],
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
                if any(cell in blocked_cells for cell in cells):
                    continue
                if any(int(region[cr, cc]) < 0 for cr, cc in cells):
                    continue
                if cells & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue
                if _t_neighborhood((r, c), stem) & occupied_cells:
                    continue

                ok_source = True
                for bd in bar:
                    rr = r + stem[0] + bd[0]
                    cc = c + stem[1] + bd[1]
                    if not _in_bounds(region.shape, rr, cc) or int(region[rr, cc]) != src:
                        ok_source = False
                        break
                if not ok_source:
                    continue

                step = 2
                found = False
                while True:
                    nr, nc = r + stem[0] * step, c + stem[1] * step
                    if not _in_bounds(region.shape, nr, nc):
                        break
                    if (nr, nc) in blocked_cells or (nr, nc) in occupied_cells or (nr, nc) in cells:
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
    if kind == "normal":
        cells = _t_cells(center, stem)
        occupied_cells |= cells
        occupied_neighbors |= (_t_neighborhood(center, stem) - cells)
        return

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


def _paint_region_grid(region: np.ndarray, region_ranks: list[int], palette: list[int]) -> np.ndarray:
    out = np.full(region.shape, -1, dtype=int)
    for rid, rank in enumerate(region_ranks):
        out[region == rid] = int(palette[rank])
    return out


def _stamp_palette_key(grid: np.ndarray, input_palette: list[int], output_palette: list[int]) -> None:
    r0, c0 = KEY_TOP_LEFT
    grid[r0, c0 : c0 + KEY_COLS] = np.array(input_palette, dtype=int)
    grid[r0 + 1, c0 : c0 + KEY_COLS] = np.array(output_palette, dtype=int)


def _stamp_markers_input(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
    input_palette: list[int],
) -> None:
    for kind, center, stem, _src, _dst, mix_rank in placed_markers:
        if kind == "normal":
            for r, c in _t_cells(center, stem):
                grid[r, c] = 1
            continue
        assert mix_rank is not None
        grid[center[0], center[1]] = int(input_palette[mix_rank])
        for r, c in _t_branch_cells(center, stem):
            grid[r, c] = 1


def _stamp_markers_output(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
    final_ranks: list[int],
    output_palette: list[int],
) -> None:
    for _kind, center, stem, src, _dst, _mix_rank in placed_markers:
        color = int(output_palette[final_ranks[src]])
        for r, c in _t_cells(center, stem):
            grid[r, c] = color


def _try_place_l9(
    region: np.ndarray,
    forbidden: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    final_ranks: list[int],
    output_palette: list[int],
) -> str | None:
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape
    candidates: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            cells = [(r + dr, c + dc) for dr, dc in offsets]
            if not all(_in_bounds(region.shape, rr, cc) for rr, cc in cells):
                continue
            if any((rr, cc) in forbidden for rr, cc in cells):
                continue
            rid = int(region[r, c])
            if rid < 0:
                continue
            if not all(int(region[rr, cc]) == rid for rr, cc in cells):
                continue
            candidates.append((r, c, rid))
    if not candidates:
        return None
    lr, lc, rid = random.choice(candidates)
    for dr, dc in offsets:
        rr, cc = lr + dr, lc + dc
        input_grid[rr, cc] = 9
        output_grid[rr, cc] = int(output_palette[final_ranks[rid]])
    return orient


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(max(GRID_SIZE_MIN, int(REGION_COUNT * REGION_SIZE_FACTOR)), GRID_SIZE_MAX)
    W = random.randint(max(GRID_SIZE_MIN, int(REGION_COUNT * REGION_SIZE_FACTOR)), GRID_SIZE_MAX)

    key_cells = _palette_key_cells()
    region = _grow_regions(H, W, REGION_COUNT, key_cells)
    if region is None:
        return None

    region_ranks = _choose_region_ranks(REGION_COUNT)
    marker_defs = _choose_marker_defs(_find_adjacent_pairs(region), region_ranks)
    if marker_defs is None:
        return None

    colors = random.sample(COLOR_POOL, 6)
    input_palette = colors[:PALETTE_SIZE]
    output_palette = colors[PALETTE_SIZE:]

    blocked_for_objects = set(key_cells) | _palette_key_buffer(region.shape)
    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]] = []
    for kind, src, dst, mix_rank in marker_defs:
        cands = _find_valid_placements(
            region,
            src,
            dst,
            blocked_for_objects,
            occupied_cells,
            occupied_neighbors,
        )
        if not cands:
            return None
        center, stem = random.choice(cands)
        _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((kind, center, stem, src, dst, mix_rank))

    final_ranks = list(region_ranks)
    saw_normal_nontrivial = False
    saw_colored_nontrivial = False
    for kind, _center, _stem, src, dst, mix_rank in placed_markers:
        if kind == "normal":
            mix_operand = int(region_ranks[src])
            if abs(region_ranks[src] - region_ranks[dst]) == 2:
                saw_normal_nontrivial = True
        else:
            assert mix_rank is not None
            mix_operand = int(mix_rank)
            if abs(mix_rank - region_ranks[dst]) == 2:
                saw_colored_nontrivial = True
        final_ranks[dst] = (mix_operand + region_ranks[dst]) // 2
    if not (saw_normal_nontrivial and saw_colored_nontrivial):
        return None

    input_grid = _paint_region_grid(region, region_ranks, input_palette)
    output_grid = _paint_region_grid(region, final_ranks, output_palette)
    _stamp_palette_key(input_grid, input_palette, output_palette)
    _stamp_palette_key(output_grid, input_palette, output_palette)
    _stamp_markers_input(input_grid, placed_markers, input_palette)
    _stamp_markers_output(output_grid, placed_markers, final_ranks, output_palette)

    l9_forbidden = blocked_for_objects | occupied_cells | occupied_neighbors
    l9_orient = _try_place_l9(
        region,
        l9_forbidden,
        input_grid,
        output_grid,
        final_ranks,
        output_palette,
    )
    if l9_orient == "top_right":
        output_grid = np.rot90(output_grid, 1)
    elif l9_orient == "top_left":
        output_grid = np.rot90(output_grid, 3)

    try:
        solved = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(solved, output_grid):
        return None
    return input_grid, output_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
