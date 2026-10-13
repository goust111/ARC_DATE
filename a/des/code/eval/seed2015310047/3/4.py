from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
MAROON = 9

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_VISIBLE_COLORS = (1, 2, 3, 6, 7, 8)
GEN_REGION_COLORS = (BLACK,) + GEN_VISIBLE_COLORS
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _label_regions(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    labels = np.full(grid.shape, -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(grid != GREY):
        rid = len(regions)
        for r, c in comp:
            labels[r, c] = rid
        regions.append(comp)
    return labels, regions


def _region_fill_color(grid: np.ndarray, cells: list[tuple[int, int]]) -> int:
    colors = {int(grid[r, c]) for r, c in cells if int(grid[r, c]) != MAROON}
    assert len(colors) == 1, f"expected one fill color per region, got {colors}"
    return int(next(iter(colors)))


def _arrow_info(grid: np.ndarray, cells: list[tuple[int, int]]) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    arrow = [(int(r), int(c)) for r, c in cells if int(grid[r, c]) == MAROON]
    if not arrow:
        return None
    assert len(arrow) == 3, f"expected a 3-pixel arrow, got {arrow}"
    arrow_set = set(arrow)
    deg = {}
    for r, c in arrow:
        deg[(r, c)] = sum((r + dr, c + dc) in arrow_set for dr, dc in DIRS4)
    corners = [pt for pt, d in deg.items() if d == 2]
    ends = [pt for pt, d in deg.items() if d == 1]
    assert len(corners) == 1 and len(ends) == 2, f"invalid arrow shape: {arrow}"
    return corners[0], ends


def _trace_target(
    grid: np.ndarray,
    labels: np.ndarray,
    source: int,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    dr, dc = direction
    r, c = start
    dist = 0
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if int(labels[nr, nc]) == int(source):
            r, c = nr, nc
            dist += 1
            continue
        if int(grid[nr, nc]) != GREY:
            target = int(labels[nr, nc])
            return None if target == int(source) or target < 0 else (target, dist + 1)
        tr, tc = nr + dr, nc + dc
        if not (0 <= tr < h and 0 <= tc < w):
            return None
        target = int(labels[tr, tc])
        if int(grid[tr, tc]) == GREY or target == int(source) or target < 0:
            return None
        return target, dist + 1


def _decode_arrow_target(
    grid: np.ndarray,
    labels: np.ndarray,
    source: int,
    corner: tuple[int, int],
    ends: list[tuple[int, int]],
) -> tuple[int, tuple[int, int]]:
    candidates: list[tuple[int, int, tuple[int, int]]] = []
    for er, ec in ends:
        direction = (corner[0] - er, corner[1] - ec)
        traced = _trace_target(grid, labels, source, corner, direction)
        if traced is None:
            continue
        target, dist = traced
        candidates.append((dist, target, direction))
    assert candidates, f"arrow in region {source} points nowhere valid"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][0] == best[0] and candidates[1][1] != best[1]:
        raise AssertionError(f"ambiguous arrow in region {source}: {candidates}")
    return int(best[1]), best[2]


def _analyze(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]], dict[int, int], list[int], dict[int, list[int]]]:
    labels, regions = _label_regions(grid)
    fill_colors = {rid: _region_fill_color(grid, cells) for rid, cells in enumerate(regions)}
    sources: list[int] = []
    pushes: dict[int, list[int]] = {}
    for rid, cells in enumerate(regions):
        info = _arrow_info(grid, cells)
        if info is None:
            continue
        corner, ends = info
        target, _direction = _decode_arrow_target(grid, labels, rid, corner, ends)
        sources.append(rid)
        pushes.setdefault(target, []).append(fill_colors[rid])
    return labels, regions, fill_colors, sources, pushes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each arrow region's color through the pointed grey wall into the adjacent region.

    If one target gets one unique pushed color, it becomes that color.
    If it gets multiple different pushed colors, it becomes yellow.
    Source regions become black and all arrows disappear.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _labels, regions, fill_colors, sources, pushes = _analyze(g)

    final_colors = dict(fill_colors)
    for target, incoming in pushes.items():
        uniq = set(map(int, incoming))
        final_colors[target] = int(next(iter(uniq))) if len(uniq) == 1 else YELLOW
    for source in sources:
        final_colors[source] = BLACK

    for rid, cells in enumerate(regions):
        color = int(final_colors[rid])
        for r, c in cells:
            out[r, c] = color
    out[g == GREY] = GREY
    return out


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _random_partition(rows: int, cols: int) -> list[list[tuple[int, int]]]:
    unassigned = {(r, c) for r in range(rows) for c in range(cols)}
    groups: list[list[tuple[int, int]]] = []
    size_choices = [1, 2, 2, 3, 3, 4]
    while unassigned:
        seed = random.choice(tuple(unassigned))
        target_size = random.choice(size_choices)
        group = [seed]
        unassigned.remove(seed)
        while len(group) < target_size:
            frontier = []
            for cell in group:
                for nbr in _coarse_neighbors(cell, rows, cols):
                    if nbr in unassigned:
                        frontier.append(nbr)
            if not frontier:
                break
            pick = random.choice(frontier)
            group.append(pick)
            unassigned.remove(pick)
        groups.append(group)
    return groups


def _edges_from_sizes(sizes: list[int]) -> list[int]:
    edges = [0]
    cur = 0
    for size in sizes:
        cur += int(size) + 1
        edges.append(cur)
    return edges


def _owner_from_partition(
    row_sizes: list[int],
    col_sizes: list[int],
    groups: list[list[tuple[int, int]]],
) -> np.ndarray:
    row_edges = _edges_from_sizes(row_sizes)
    col_edges = _edges_from_sizes(col_sizes)
    h = row_edges[-1] + 1
    w = col_edges[-1] + 1
    owner = np.full((h, w), -1, dtype=int)
    group_sets = [set(group) for group in groups]

    for rid, group in enumerate(groups):
        cells = group_sets[rid]
        for gr, gc in group:
            rr = slice(row_edges[gr] + 1, row_edges[gr + 1])
            cc = slice(col_edges[gc] + 1, col_edges[gc + 1])
            owner[rr, cc] = rid
            if (gr + 1, gc) in cells:
                wall_r = row_edges[gr + 1]
                owner[wall_r, col_edges[gc] + 1 : col_edges[gc + 1]] = rid
            if (gr, gc + 1) in cells:
                wall_c = col_edges[gc + 1]
                owner[row_edges[gr] + 1 : row_edges[gr + 1], wall_c] = rid
    return owner


def _trace_owner(
    owner: np.ndarray,
    source: int,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = owner.shape
    dr, dc = direction
    r, c = start
    dist = 0
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if int(owner[nr, nc]) == int(source):
            r, c = nr, nc
            dist += 1
            continue
        if int(owner[nr, nc]) >= 0:
            target = int(owner[nr, nc])
            return None if target == int(source) else (target, dist + 1)
        tr, tc = nr + dr, nc + dc
        if not (0 <= tr < h and 0 <= tc < w) or int(owner[tr, tc]) < 0:
            return None
        target = int(owner[tr, tc])
        return None if target == int(source) else (target, dist + 1)


def _enumerate_arrow_candidates(owner: np.ndarray) -> dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]:
    by_pair: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] = {}
    n_regions = int(owner.max()) + 1
    region_cells = {
        rid: [(int(r), int(c)) for r, c in np.argwhere(owner == rid)]
        for rid in range(n_regions)
    }

    for source, cells in region_cells.items():
        if len(cells) < 5:
            continue
        cell_set = set(cells)
        for corner in cells:
            cr, cc = corner
            for dr, dc in DIRS4:
                head = (cr - dr, cc - dc)
                if head not in cell_set:
                    continue
                traced = _trace_owner(owner, source, corner, (dr, dc))
                if traced is None:
                    continue
                target, dist = traced
                if dist > 3:
                    continue
                for pr, pc in _perpendiculars((dr, dc)):
                    tail = (cr + pr, cc + pc)
                    if tail not in cell_set:
                        continue
                    other = _trace_owner(owner, source, corner, (-pr, -pc))
                    if other is not None and other[1] <= dist:
                        continue
                    by_pair.setdefault((source, target), []).append((head, corner, tail))
    return by_pair


def _choose_plan(
    placements: dict[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]
) -> list[dict[str, object]] | None:
    pairs = list(placements.keys())
    by_target: dict[int, list[int]] = {}
    for source, target in pairs:
        by_target.setdefault(int(target), []).append(int(source))

    mix_choices = []
    for target, sources in by_target.items():
        uniq_sources = sorted(set(map(int, sources)))
        for s1, s2 in combinations(uniq_sources, 2):
            mix_choices.append((target, s1, s2))
    random.shuffle(mix_choices)

    for mix_target, s1, s2 in mix_choices:
        used_sources = {int(s1), int(s2)}
        used_targets = {int(mix_target)}
        blocked = used_sources | used_targets
        single_options = [
            (int(source), int(target))
            for source, target in pairs
            if int(source) not in blocked and int(target) not in blocked
        ]
        random.shuffle(single_options)
        if not single_options:
            continue

        arrows = [
            {"source": int(s1), "target": int(mix_target)},
            {"source": int(s2), "target": int(mix_target)},
        ]

        s3, t3 = single_options[0]
        arrows.append({"source": int(s3), "target": int(t3)})
        used_sources.add(int(s3))
        used_targets.add(int(t3))

        blocked = used_sources | used_targets
        extra_options = [
            (int(source), int(target))
            for source, target in pairs
            if int(source) not in blocked and int(target) not in blocked
        ]
        if extra_options and random.random() < 0.5:
            s4, t4 = random.choice(extra_options)
            arrows.append({"source": int(s4), "target": int(t4)})

        for arrow in arrows:
            arrow["shape"] = random.choice(placements[(int(arrow["source"]), int(arrow["target"]))])
        return arrows
    return None


def _assign_colors(n_regions: int, arrows: list[dict[str, object]]) -> dict[int, int]:
    colors: dict[int, int] = {}
    incoming: dict[int, list[int]] = {}
    for arrow in arrows:
        incoming.setdefault(int(arrow["target"]), []).append(int(arrow["source"]))

    for target, sources in incoming.items():
        if len(sources) > 1:
            chosen = random.sample(GEN_VISIBLE_COLORS, len(sources))
            for source, color in zip(sources, chosen, strict=False):
                colors[int(source)] = int(color)

    for arrow in arrows:
        source = int(arrow["source"])
        if source not in colors:
            colors[source] = int(random.choice(GEN_VISIBLE_COLORS))

    for target, sources in incoming.items():
        pushed = {colors[int(source)] for source in sources}
        forbidden = pushed if len(pushed) == 1 else {YELLOW}
        choices = [color for color in GEN_REGION_COLORS if int(color) not in forbidden]
        colors[int(target)] = int(random.choice(choices))

    for rid in range(n_regions):
        if rid in colors:
            continue
        if random.random() < 0.18:
            colors[rid] = BLACK
        else:
            colors[rid] = int(random.choice(GEN_VISIBLE_COLORS))
    return colors


def _paint_input(owner: np.ndarray, colors: dict[int, int], arrows: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(owner.shape, GREY, dtype=int)
    for rid, color in colors.items():
        grid[owner == int(rid)] = int(color)
    for arrow in arrows:
        for r, c in arrow["shape"]:
            grid[int(r), int(c)] = MAROON
    return grid


def _expected_output(owner: np.ndarray, colors: dict[int, int], arrows: list[dict[str, object]]) -> np.ndarray:
    final_colors = dict(colors)
    pushed: dict[int, list[int]] = {}
    for arrow in arrows:
        source = int(arrow["source"])
        target = int(arrow["target"])
        pushed.setdefault(target, []).append(int(colors[source]))
    for target, incoming in pushed.items():
        uniq = set(incoming)
        final_colors[target] = int(next(iter(uniq))) if len(uniq) == 1 else YELLOW
    for arrow in arrows:
        final_colors[int(arrow["source"])] = BLACK

    out = np.full(owner.shape, GREY, dtype=int)
    for rid, color in final_colors.items():
        out[owner == int(rid)] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    coarse_rows = random.randint(3, 5)
    coarse_cols = random.randint(3, 5)
    row_sizes = [random.randint(2, 4) for _ in range(coarse_rows)]
    col_sizes = [random.randint(2, 4) for _ in range(coarse_cols)]
    groups = _random_partition(coarse_rows, coarse_cols)
    if len(groups) < 5:
        return None

    owner = _owner_from_partition(row_sizes, col_sizes, groups)
    placements = _enumerate_arrow_candidates(owner)
    if not placements:
        return None

    arrows = _choose_plan(placements)
    if arrows is None:
        return None

    n_regions = int(owner.max()) + 1
    colors = _assign_colors(n_regions, arrows)
    inp = _paint_input(owner, colors, arrows)
    expected = _expected_output(owner, colors, arrows)

    produced = main(inp)
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
