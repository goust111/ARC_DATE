# Combo variant of 21897d95 (index 25)
# combo_spec: D1/2+D5/2+D10/2
# Name: 朝向分步反向流

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis:
# a) Concrete behavior trace:
#    - All T controls are now the same 4-cell color-1 tetromino. Shape no longer tells
#      us whether the marker is "normal" or "colored"; absolute orientation does.
#    - Vertical stems (up/down) are normal markers. Horizontal stems (left/right) are
#      colored markers.
#    - D1 flips the color flow for both families: the stem still points at a target-side
#      donor region, but the region that actually changes is the source side identified by
#      the two diagonals behind the stem.
#    - For a vertical T, phase behavior is: source region -> target region's original
#      color. For a horizontal T, source region -> unique third touched region's original
#      color. The pointed-at target stays unchanged in both cases.
#    - D10 makes this sequential: first apply every vertical T simultaneously, then parse
#      the horizontal T controls again on the modified grid and apply them simultaneously.
#      Example: if vertical B->A runs in phase 1, B becomes A-colored and merges into A.
#      If a later horizontal T uses A as its source side, phase 2 recolors (A union B),
#      whereas a simultaneous solver would recolor only A.
# b) Complexity calibration:
#    - The generator always includes one horizontal T and several vertical Ts, so users
#      can compare the same symbol under different orientations.
#    - Horizontal markers are only placed where they touch exactly three regions:
#      source, pointed-at target, and a visible third region on the open back side.
#      Vertical markers are restricted to touch exactly two regions, keeping the contrast
#      legible instead of cluttered.
#    - The sequential effect is not left to chance: every generated grid contains a
#      deliberate vertical->horizontal interaction chain that changes the phase-2 source.
# c) Hidden-assumption audit:
#    - The original solver assumed colored markers were detected by a non-1 center cell;
#      that is invalid here because every T is a full 4-cell color-1 tetromino.
#    - The original simultaneous DSU/update pass is also invalid because phase-2 source
#      and third-region roots must be read from the grid after phase 1.
#    - D1 moves the write target from destination root to source root, so conflict checks
#      must also move from destination to source.
#    - Horizontal replacement can no longer come from a stored center value; it must be
#      inferred from touched components, with exactly one extra root besides source/dest.
# d) Placement legality:
#    - Markers may not overlap or touch each other.
#    - Vertical Ts only use up/down stems and must touch exactly {source, target}.
#    - Horizontal Ts only use left/right stems and must touch exactly
#      {source, target, third}, with the back-side cell inside the third region.
#    - The generator keeps phase-1 sources unique, and uses exactly one horizontal source,
#      so the reversed writes cannot demand conflicting colors.
# e) Visual observability:
#    - Humans can read orientation directly because every control has the same shape.
#    - The third-color rule is visible in the input because the horizontal T sits at a
#      three-region contact; the extra donor is an actual neighboring color patch, not an
#      invisible metadata choice.
#    - The phase-order rule is visible because one vertical T deliberately expands the
#      horizontal source before phase 2.
# f) Visual cleanliness:
#    - Region colors exclude 1 and 9, reserving 1 for T markers and 9 for the optional
#      rotation L marker.
#    - Marker neighborhoods are buffered, so control shapes remain separated and easy to
#      parse.
# g) Color-change inferability:
#    - Input regions use distinct colors, so reverse flow is visible: the pointed-at side
#      stays put while the diagonal/source side changes.
#    - The generator rejects samples unless the sequential combo output differs from
#      three simpler baselines: simultaneous combo, no-orientation reverse flow, and
#      forward-flow sequential. This prevents silent degeneration back to easier rules.


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
    if int(stem[0]) == 0:
        return [UP, DOWN]
    return [LEFT, RIGHT]


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


def _mark_cells(mask: np.ndarray, markers: list[dict]) -> None:
    for marker in markers:
        for r, c in marker["cells"]:
            mask[int(r), int(c)] = True


def _label_base_components(
    g: np.ndarray, control: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
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


def _parse_controls(g: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        int(k): {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == int(k))}
        for k in range(1, int(n_1) + 1)
    }

    normal_markers: list[dict] = []
    colored_markers: list[dict] = []
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
            "kind": "colored" if int(stem[0]) == 0 else "normal",
            "center": center,
            "stem": stem,
            "bar": bar,
            "cells": set(cells),
        }
        next_mid += 1
        if marker["kind"] == "normal":
            normal_markers.append(marker)
        else:
            colored_markers.append(marker)
        used_ones |= cells
        _mark_cells(control, [marker])

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to 4-cell T markers"

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

    assert colored_markers, "at least one horizontal T marker is required"
    return normal_markers, colored_markers, small_markers


def _apply_controls(
    g: np.ndarray,
    current_t_markers: list[dict],
    active_t_markers: list[dict],
    small_markers: list[dict],
    fill_small: bool,
    *,
    use_orientation: bool,
    reverse_flow: bool,
) -> np.ndarray:
    control = np.zeros(g.shape, dtype=bool)
    _mark_cells(control, current_t_markers)
    _mark_cells(control, small_markers)

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"

    dsu = _DSU(len(comp_colors))
    active_ids = {int(marker["mid"]) for marker in active_t_markers}
    marker_source_comp: dict[int, int] = {}
    marker_dest_comp: dict[int, int] = {}
    marker_adjacent: dict[int, list[int]] = {}

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

        adj_counts = _adjacent_component_counts(marker["cells"], comp_id)
        for comp in adj_counts:
            if comp_colors[comp] == comp_colors[source]:
                dsu.union(source, comp)

        mid = int(marker["mid"])
        marker_source_comp[mid] = source
        if mid in active_ids:
            marker_dest_comp[mid] = int(dest)
            marker_adjacent[mid] = list(adj_counts)

    small_owner_comp: dict[int, int] = {}
    for marker in small_markers:
        counts = _adjacent_component_counts(marker["cells"], comp_id)
        assert counts, "small marker must touch its owner region"
        owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
        for comp in counts:
            if comp_colors[comp] == comp_colors[owner]:
                dsu.union(owner, comp)
        small_owner_comp[int(marker["mid"])] = int(owner)

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    final_color = dict(root_color)
    for marker in active_t_markers:
        mid = int(marker["mid"])
        source = dsu.find(marker_source_comp[mid])
        dest = dsu.find(marker_dest_comp[mid])
        if use_orientation and marker["kind"] == "colored":
            extra_roots = {
                dsu.find(comp)
                for comp in marker_adjacent[mid]
                if dsu.find(comp) not in {source, dest}
            }
            assert len(extra_roots) == 1, "horizontal T must touch exactly one third region"
            replacement = int(root_color[next(iter(extra_roots))])
        else:
            replacement = int(root_color[dest] if reverse_flow else root_color[source])

        changed = source if reverse_flow else dest
        assert final_color[changed] in {root_color[changed], replacement}, "conflicting T controls"
        final_color[changed] = replacement

    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[dsu.find(cid)])

    for marker in active_t_markers:
        source = dsu.find(marker_source_comp[int(marker["mid"])])
        fill = int(final_color[source])
        for r, c in marker["cells"]:
            out[int(r), int(c)] = fill

    if fill_small:
        for marker in small_markers:
            owner = dsu.find(small_owner_comp[int(marker["mid"])])
            fill = int(final_color[owner])
            for r, c in marker["cells"]:
                out[int(r), int(c)] = fill

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def _solve_unrotated_sequential(g: np.ndarray) -> np.ndarray:
    normal_markers, colored_markers, small_markers = _parse_controls(g)
    mid = _apply_controls(
        g,
        normal_markers + colored_markers,
        normal_markers,
        small_markers,
        fill_small=False,
        use_orientation=True,
        reverse_flow=True,
    )
    return _apply_controls(
        mid,
        colored_markers,
        colored_markers,
        small_markers,
        fill_small=True,
        use_orientation=True,
        reverse_flow=True,
    )


def _solve_unrotated_simultaneous(g: np.ndarray) -> np.ndarray:
    normal_markers, colored_markers, small_markers = _parse_controls(g)
    return _apply_controls(
        g,
        normal_markers + colored_markers,
        normal_markers + colored_markers,
        small_markers,
        fill_small=True,
        use_orientation=True,
        reverse_flow=True,
    )


def _solve_unrotated_no_orientation(g: np.ndarray) -> np.ndarray:
    normal_markers, colored_markers, small_markers = _parse_controls(g)
    mid = _apply_controls(
        g,
        normal_markers + colored_markers,
        normal_markers,
        small_markers,
        fill_small=False,
        use_orientation=False,
        reverse_flow=True,
    )
    return _apply_controls(
        mid,
        colored_markers,
        colored_markers,
        small_markers,
        fill_small=True,
        use_orientation=False,
        reverse_flow=True,
    )


def _solve_unrotated_forward(g: np.ndarray) -> np.ndarray:
    normal_markers, colored_markers, small_markers = _parse_controls(g)
    mid = _apply_controls(
        g,
        normal_markers + colored_markers,
        normal_markers,
        small_markers,
        fill_small=False,
        use_orientation=True,
        reverse_flow=False,
    )
    return _apply_controls(
        mid,
        colored_markers,
        colored_markers,
        small_markers,
        fill_small=True,
        use_orientation=True,
        reverse_flow=False,
    )


def _behavior_metrics(g: np.ndarray) -> dict[str, int]:
    seq = _solve_unrotated_sequential(g)
    sim = _solve_unrotated_simultaneous(g)
    no_orientation = _solve_unrotated_no_orientation(g)
    forward = _solve_unrotated_forward(g)
    return {
        "phase_delta": int(np.count_nonzero(seq != sim)),
        "orientation_delta": int(np.count_nonzero(seq != no_orientation)),
        "flow_delta": int(np.count_nonzero(seq != forward)),
    }


def _find_l9_orientation(g: np.ndarray) -> str | None:
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    for k in range(1, int(n) + 1):
        coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == int(k))}
        if len(coords) != 5:
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
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
    Read orientation-coded T markers, run a reversed two-phase recolor, and optionally rotate the result with a color-9 L marker.

    Input structure:
    - The grid is partitioned into irregular monochrome regions.
    - Every color-1 control is a full 4-cell T tetromino on a region boundary; there is no colored-center form.
    - Vertical-stem Ts and horizontal-stem Ts share the same shape, so absolute orientation is the only cue for which rule family they use.
    - An optional 5-cell color-9 L marker sits inside one region and still signals a quarter-turn of the finished output.

    Rule:
    - The stem still points toward a target-side region, and the two diagonals behind the stem still identify the source side that owns the marker.
    - Phase 1 applies all vertical Ts simultaneously: each source-side region is recolored to the pointed-at target region's original color, while the target itself stays unchanged.
    - Phase 2 then applies all horizontal Ts on the already modified grid: each source-side region is recolored to the original color of the unique third region touching that T, while the pointed-at target again stays unchanged.
    - After each phase, processed T cells are filled with the source side's final color; any small scraps, including the L marker, are absorbed at the end. L orientation remains ┐ -> CCW 90 degrees and ┌ -> CW 90 degrees.

    Key insight:
    - You must read both orientation and timing: vertical Ts act first and can enlarge the later horizontal source, and in both phases the region that changes is opposite the stem direction rather than the region being pointed at.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated_sequential(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (orientation-coded reversed two-phase flow) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 2, 1]

MARKER_COUNTS = [4, 5]
COLORED_COUNT = 1

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 500
COLORED_OPTION_TRIES = 80


def _adjacent_region_ids(region: np.ndarray, cells: set[tuple[int, int]]) -> set[int]:
    h, w = region.shape
    touched: set[int] = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                touched.add(int(region[nr, nc]))
    return touched


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
    h, w = region.shape
    undirected: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            for dr, dc in DIRS:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    a, b = int(region[r, c]), int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed: list[tuple[int, int]] = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _choose_marker_count(n_regions: int) -> int:
    choices = [count for count in MARKER_COUNTS if count <= n_regions]
    assert choices
    return random.choice(choices)


def _find_valid_placements(
    region: np.ndarray,
    kind: str,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], int | None]]:
    h, w = region.shape
    stems = (UP, DOWN) if kind == "normal" else (LEFT, RIGHT)
    candidates: list[tuple[tuple[int, int], tuple[int, int], int | None]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            for stem in stems:
                cells = _t_cells((r, c), stem)
                if cells & occupied_cells:
                    continue
                if _t_neighborhood((r, c), stem) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue

                bar = _get_bar(stem)
                ok_source = True
                for bd in bar:
                    rr = int(r + stem[0] + bd[0])
                    cc = int(c + stem[1] + bd[1])
                    if not (0 <= rr < h and 0 <= cc < w and int(region[rr, cc]) == int(src)):
                        ok_source = False
                        break
                if not ok_source:
                    continue

                step = 2
                found = False
                while True:
                    nr, nc = int(r + stem[0] * step), int(c + stem[1] * step)
                    if not (0 <= nr < h and 0 <= nc < w):
                        break
                    if (nr, nc) in occupied_cells or (nr, nc) in cells:
                        step += 1
                        continue
                    found = int(region[nr, nc]) == int(dst)
                    break
                if not found:
                    continue

                touched = _adjacent_region_ids(region, cells)
                if kind == "normal":
                    if touched == {int(src), int(dst)}:
                        candidates.append(((int(r), int(c)), stem, None))
                    continue

                back_r, back_c = int(r - stem[0]), int(c - stem[1])
                if not (0 <= back_r < h and 0 <= back_c < w):
                    continue
                third = int(region[back_r, back_c])
                if third in {int(src), int(dst)}:
                    continue
                if touched == {int(src), int(dst), int(third)}:
                    candidates.append(((int(r), int(c)), stem, int(third)))
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
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
) -> None:
    for _kind, center, stem, _src, _dst, _third in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[int(cr), int(cc)] = 1


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
    input_grid: np.ndarray,
) -> None:
    if random.random() >= L9_PROBABILITY:
        return
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    h, w = region.shape

    blocked: set[tuple[int, int]] = set()
    for _kind, center, stem, *_ in placed_markers:
        blocked |= _t_cells(center, stem)
        blocked |= _t_neighborhood(center, stem)

    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            cells = [(int(r + dr), int(c + dc)) for dr, dc in offsets]
            if not all(0 <= cr < h and 0 <= cc < w for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if any((cr, cc) in blocked for cr, cc in cells):
                continue
            candidates.append((int(r), int(c)))

    if not candidates:
        return
    lr, lc = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[int(lr + dr), int(lc + dc)] = 9


def _place_markers(
    region: np.ndarray, n_markers: int
) -> list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]] | None:
    directed_pairs = _find_adjacent_pairs(region)
    colored_options: list[tuple[int, int, int, tuple[int, int], tuple[int, int]]] = []
    for src, dst in directed_pairs:
        for center, stem, third in _find_valid_placements(region, "colored", src, dst, set(), set()):
            assert third is not None
            colored_options.append((int(src), int(dst), int(third), center, stem))
    if not colored_options:
        return None

    random.shuffle(colored_options)
    for src_a, dst_c, third_x, center, stem in colored_options[:COLORED_OPTION_TRIES]:
        occupied_cells: set[tuple[int, int]] = set()
        occupied_neighbors: set[tuple[int, int]] = set()
        placed: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]] = []
        used_sources = {int(src_a)}

        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        placed.append(("colored", center, stem, int(src_a), int(dst_c), int(third_x)))

        interaction_options = []
        for src_b, dst in directed_pairs:
            if int(dst) != int(src_a):
                continue
            if int(src_b) in {int(src_a), int(dst_c), int(third_x)}:
                continue
            cands = _find_valid_placements(region, "normal", int(src_b), int(src_a), occupied_cells, occupied_neighbors)
            if cands:
                interaction_options.append((len(cands), random.random(), int(src_b), cands))
        if not interaction_options:
            continue

        interaction_options.sort(key=lambda item: (item[0], item[1]))
        _, _, src_b, cands = interaction_options[0]
        center_b, stem_b, _ = random.choice(cands)
        _register_marker(center_b, stem_b, occupied_cells, occupied_neighbors)
        placed.append(("normal", center_b, stem_b, int(src_b), int(src_a), None))
        used_sources.add(int(src_b))

        remaining_normals = n_markers - COLORED_COUNT - 1
        ok = True
        for _ in range(remaining_normals):
            options = []
            for src, dst in directed_pairs:
                if int(src) in used_sources:
                    continue
                if int(dst) == int(src_a):
                    continue
                cands = _find_valid_placements(region, "normal", int(src), int(dst), occupied_cells, occupied_neighbors)
                if cands:
                    options.append((len(cands), random.random(), int(src), int(dst), cands))
            if not options:
                ok = False
                break
            options.sort(key=lambda item: (item[0], item[1]))
            _, _, src, dst, cands = random.choice(options[: min(5, len(options))])
            center_n, stem_n, _ = random.choice(cands)
            _register_marker(center_n, stem_n, occupied_cells, occupied_neighbors)
            placed.append(("normal", center_n, stem_n, int(src), int(dst), None))
            used_sources.add(int(src))
        if ok:
            return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    n_markers = _choose_marker_count(n_regions)
    placed_markers = _place_markers(region, n_markers)
    if placed_markers is None:
        return None

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    input_grid = _build_color_map(region, input_colors)
    _stamp_markers_input(input_grid, placed_markers)
    _try_place_l9(region, placed_markers, input_grid)

    try:
        metrics = _behavior_metrics(input_grid)
        _ = main(input_grid)
    except Exception:
        return None

    if metrics["phase_delta"] < MIN_REGION_CELLS:
        return None
    if metrics["orientation_delta"] < MIN_REGION_CELLS:
        return None
    if metrics["flow_delta"] < MIN_REGION_CELLS:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
