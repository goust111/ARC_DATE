from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Ray:
    color: int
    marker: tuple[int, int]
    ray_dir: tuple[int, int]
    component_cells: tuple[tuple[int, int], ...]
    hit_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Placement:
    marker: tuple[int, int]
    ray_dir: tuple[int, int]
    cells: tuple[tuple[int, int], ...]
    trail_len: int
    gap: int


@dataclass(frozen=True)
class Candidate:
    axis: str
    line_value: int
    side: str
    segment_count: int
    hit_cells: tuple[tuple[int, int], ...]
    placements: tuple[Placement, ...]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _count_segments(vals: list[int]) -> int:
    if not vals:
        return 0
    runs = 1
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1] + 1:
            runs += 1
    return runs


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _line_component_info(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]]:
    assert 3 <= len(coords) <= 4, "each colored marker+trail component must contain 3 or 4 cells"
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        row = rows[0]
        ordered = [(row, c) for c in sorted(c for _, c in coords)]
        assert [c for _, c in ordered] == list(range(ordered[0][1], ordered[-1][1] + 1))
    else:
        assert len(cols) == 1, "marker/trail component must be a straight horizontal or vertical line"
        col = cols[0]
        ordered = [(r, col) for r, _ in sorted(coords)]
        assert [r for r, _ in ordered] == list(range(ordered[0][0], ordered[-1][0] + 1))
    return ordered, (ordered[0], ordered[-1])


def _collect_grey_hits(grid: np.ndarray, marker: tuple[int, int], ray_dir: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    hits: list[tuple[int, int]] = []
    r = marker[0] + ray_dir[0]
    c = marker[1] + ray_dir[1]
    while in_bounds(grid.shape, r, c):
        val = int(grid[r, c])
        if val == GREY:
            hits.append((r, c))
        elif val != BLACK:
            break
        r += ray_dir[0]
        c += ray_dir[1]
    return tuple(hits)


def _identify_rays(grid: np.ndarray) -> list[Ray]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(int(v) for v in np.unique(g)) - {BLACK, GREY})
    assert 3 <= len(colors) <= 4, "expected 3 or 4 distinct marker colors"

    rays: list[Ray] = []
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        ordered, endpoints = _line_component_info(coords)

        candidates: list[Ray] = []
        for endpoint in endpoints:
            if endpoint == ordered[0]:
                neighbor = ordered[1]
            else:
                neighbor = ordered[-2]
            trail_dir = (neighbor[0] - endpoint[0], neighbor[1] - endpoint[1])
            ray_dir = (-trail_dir[0], -trail_dir[1])
            hits = _collect_grey_hits(g, endpoint, ray_dir)
            if hits:
                candidates.append(
                    Ray(
                        color=color,
                        marker=endpoint,
                        ray_dir=ray_dir,
                        component_cells=tuple(ordered),
                        hit_cells=hits,
                    )
                )

        assert len(candidates) == 1, "each color must contain one uniquely oriented marker+trail component"
        rays.append(candidates[0])

    rays.sort(key=lambda ray: (ray.marker[0], ray.marker[1]))
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """Color the grey mask only where reverse-cast rays pass through it, then erase the markers and trails.

    Input structure:
    - A black background contains one irregular grey mask and 3-4 detached colored marker+trail components.
    - Each colored component is a straight 3-4 pixel horizontal or vertical line in a unique color from blue/red/green/yellow.
    - The marker is the endpoint that faces the grey mask; the same-color trail extends away from the mask.

    Rule:
    - For each colored component, infer the trail direction from the endpoint toward the rest of that straight line.
    - Cast a ray from the marker in the opposite direction, using the original input grid as the obstacle map.
    - Along that ray, recolor every grey mask pixel to the marker color, skip black cells, and stop only when a non-black non-grey pixel or the grid edge is reached.
    - Process rays in reading order of their marker positions, so later rays overwrite earlier colors on shared grey pixels.
    - Finally erase all original marker and trail pixels to black, while untouched grey mask pixels remain grey.

    Key insight:
    - Grey is not the ray itself and not a keep-mask for existing colors; it is the only region that can be recolored as the reverse-cast rays pass through it, even across black gaps between separate grey segments on the same row or column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rays = _identify_rays(g)
    out = g.copy()
    for ray in rays:
        for r, c in ray.hit_cells:
            out[r, c] = ray.color

    out[(g != BLACK) & (g != GREY)] = BLACK
    return out


def _occupied_neighbor_count(cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((nr, nc) in cells for nr, nc in neighbors4(r, c))


def _degree_map(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {cell: _occupied_neighbor_count(cells, cell) for cell in cells}


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = [next(iter(seen))]
    while stack:
        r, c = stack.pop()
        for nbr in neighbors4(r, c):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return seen == cells


def _tree_like(cells: set[tuple[int, int]]) -> bool:
    if not _connected(cells):
        return False
    edges = 0
    for r, c in cells:
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) in cells:
                edges += 1
    return edges == len(cells) - 1


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _attach_polyline(
    cells: set[tuple[int, int]],
    points: list[tuple[int, int]],
    box_h: int,
    box_w: int,
) -> bool:
    if len(points) < 2:
        return False
    added: list[tuple[int, int]] = []
    occ = set(cells)
    cur = points[0]
    for nxt_point in points[1:]:
        dr = int(np.sign(nxt_point[0] - cur[0]))
        dc = int(np.sign(nxt_point[1] - cur[1]))
        assert dr == 0 or dc == 0
        steps = abs(nxt_point[0] - cur[0]) + abs(nxt_point[1] - cur[1])
        for _ in range(steps):
            cur = (cur[0] + dr, cur[1] + dc)
            if not (0 <= cur[0] < box_h and 0 <= cur[1] < box_w):
                return False
            if cur in occ:
                return False
            if _occupied_neighbor_count(occ, cur) != 1:
                return False
            occ.add(cur)
            added.append(cur)
    cells.update(added)
    return True


def _propose_extension(
    rng: np.random.Generator,
    cells: set[tuple[int, int]],
    endpoint: tuple[int, int],
    length: int,
    box_h: int,
    box_w: int,
) -> list[tuple[int, int]] | None:
    deg = _degree_map(cells)
    assert deg[endpoint] == 1
    parent = next(nbr for nbr in neighbors4(*endpoint) if nbr in cells)
    prev = parent
    cur = endpoint
    occ = set(cells)
    path: list[tuple[int, int]] = []

    for _ in range(length):
        dirs = list(DIR4)
        rng.shuffle(dirs)
        straight = (cur[0] - prev[0], cur[1] - prev[1])
        scored: list[tuple[int, tuple[int, int]]] = []
        for dr, dc in dirs:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt == prev or nxt in occ:
                continue
            if not (0 <= nxt[0] < box_h and 0 <= nxt[1] < box_w):
                continue
            if _occupied_neighbor_count(occ, nxt) != 1:
                continue
            score = 0
            if (dr, dc) != straight:
                score += 2
            if nxt[0] in {0, box_h - 1} or nxt[1] in {0, box_w - 1}:
                score -= 1
            scored.append((score, nxt))
        if not scored:
            return None
        best_score = max(score for score, _ in scored)
        options = [nxt for score, nxt in scored if score == best_score]
        nxt = options[int(rng.integers(0, len(options)))]
        path.append(nxt)
        occ.add(nxt)
        prev, cur = cur, nxt

    return path


def _build_mask(rng: np.random.Generator) -> set[tuple[int, int]] | None:
    box_h = int(rng.integers(8, 11))
    box_w = int(rng.integers(8, 11))
    hub_r = int(rng.integers(2, box_h - 2))
    hub_c = int(rng.integers(2, box_w - 2))

    up = int(rng.integers(1, min(3, hub_r + 1)))
    down = int(rng.integers(1, min(3, box_h - hub_r)))
    left = int(rng.integers(1, min(3, hub_c + 1)))
    right = int(rng.integers(1, min(3, box_w - hub_c)))
    if min(up, down, left, right) < 1:
        return None

    cells = {(hub_r, hub_c)}
    for d in range(1, up + 1):
        cells.add((hub_r - d, hub_c))
    for d in range(1, down + 1):
        cells.add((hub_r + d, hub_c))
    for d in range(1, left + 1):
        cells.add((hub_r, hub_c - d))
    for d in range(1, right + 1):
        cells.add((hub_r, hub_c + d))

    branch_options: list[list[tuple[int, int]]] = []

    top = (hub_r - up, hub_c)
    bottom = (hub_r + down, hub_c)
    left_end = (hub_r, hub_c - left)
    right_end = (hub_r, hub_c + right)

    min_far_right = hub_c + right + 2
    if min_far_right < box_w:
        for target_c in range(min_far_right, box_w):
            branch_options.append([top, (top[0], target_c), (hub_r, target_c)])
            branch_options.append([bottom, (bottom[0], target_c), (hub_r, target_c)])
    max_far_left = hub_c - left - 2
    if max_far_left >= 0:
        for target_c in range(0, max_far_left + 1):
            branch_options.append([top, (top[0], target_c), (hub_r, target_c)])
            branch_options.append([bottom, (bottom[0], target_c), (hub_r, target_c)])

    min_far_down = hub_r + down + 2
    if min_far_down < box_h:
        for target_r in range(min_far_down, box_h):
            branch_options.append([left_end, (target_r, left_end[1]), (target_r, hub_c)])
            branch_options.append([right_end, (target_r, right_end[1]), (target_r, hub_c)])
    max_far_up = hub_r - up - 2
    if max_far_up >= 0:
        for target_r in range(0, max_far_up + 1):
            branch_options.append([left_end, (target_r, left_end[1]), (target_r, hub_c)])
            branch_options.append([right_end, (target_r, right_end[1]), (target_r, hub_c)])

    if not branch_options:
        return None
    rng.shuffle(branch_options)

    branch_added = False
    for polyline in branch_options:
        trial = set(cells)
        if _attach_polyline(trial, polyline, box_h, box_w):
            cells = trial
            branch_added = True
            break
    if not branch_added:
        return None

    target_area = int(rng.integers(17, 24))
    extension_tries = 0
    while len(cells) < target_area and extension_tries < 250:
        extension_tries += 1
        deg = _degree_map(cells)
        endpoints = [cell for cell, degree in deg.items() if degree == 1]
        rng.shuffle(endpoints)
        extended = False
        for endpoint in endpoints:
            max_len = min(4, target_area - len(cells))
            lens = list(range(max_len, 0, -1))
            rng.shuffle(lens)
            for length in lens:
                path = _propose_extension(rng, cells, endpoint, length, box_h, box_w)
                if path is None:
                    continue
                for cell in path:
                    cells.add(cell)
                extended = True
                break
            if extended:
                break
        if not extended:
            break

    if not (15 <= len(cells) <= 25):
        return None
    if not _tree_like(cells):
        return None
    normalized = _normalize_cells(cells)
    r0, c0, r1, c1 = _bbox(normalized)
    if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
        return None
    return normalized


def _embed_mask(rng: np.random.Generator, local_cells: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = _bbox(local_cells)
    mask_h = r1 - r0 + 1
    mask_w = c1 - c0 + 1
    min_h = max(13, mask_h + 7)
    min_w = max(13, mask_w + 7)
    if min_h > 18 or min_w > 18:
        return None

    height = int(rng.integers(min_h, 19))
    width = int(rng.integers(min_w, 19))

    top_min = 3
    top_max = height - mask_h - 3
    left_min = 3
    left_max = width - mask_w - 3
    if top_min > top_max or left_min > left_max:
        return None

    top = int(rng.integers(top_min, top_max + 1))
    left = int(rng.integers(left_min, left_max + 1))
    return {(r + top, c + left) for r, c in local_cells}


def _touches_mask(mask_set: set[tuple[int, int]], cell: tuple[int, int]) -> bool:
    return any(nbr in mask_set for nbr in neighbors4(*cell))


def _candidate_placements(
    mask_set: set[tuple[int, int]],
    shape: tuple[int, int],
    axis: str,
    line_value: int,
    side: str,
    near_index: int,
) -> tuple[Placement, ...]:
    placements: list[Placement] = []
    height, width = shape
    for trail_len in (2, 3):
        for gap in (2, 3, 4):
            if axis == "h" and side == "left":
                marker = (line_value, near_index - gap)
                cells = tuple((line_value, marker[1] - d) for d in range(trail_len, 0, -1)) + (marker,)
                ray_dir = (0, 1)
            elif axis == "h" and side == "right":
                marker = (line_value, near_index + gap)
                cells = (marker,) + tuple((line_value, marker[1] + d) for d in range(1, trail_len + 1))
                ray_dir = (0, -1)
            elif axis == "v" and side == "top":
                marker = (near_index - gap, line_value)
                cells = tuple((marker[0] - d, line_value) for d in range(trail_len, 0, -1)) + (marker,)
                ray_dir = (1, 0)
            else:
                marker = (near_index + gap, line_value)
                cells = (marker,) + tuple((marker[0] + d, line_value) for d in range(1, trail_len + 1))
                ray_dir = (-1, 0)

            if not all(in_bounds(shape, r, c) for r, c in cells):
                continue
            if any((r, c) in mask_set for r, c in cells):
                continue
            if any(_touches_mask(mask_set, cell) for cell in cells):
                continue
            placements.append(
                Placement(
                    marker=marker,
                    ray_dir=ray_dir,
                    cells=cells,
                    trail_len=trail_len,
                    gap=gap,
                )
            )
    return tuple(placements)


def _enumerate_candidates(mask_set: set[tuple[int, int]], shape: tuple[int, int]) -> list[Candidate]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in mask_set:
        rows[r].append(c)
        cols[c].append(r)

    candidates: list[Candidate] = []
    for row, row_cols in rows.items():
        ordered = sorted(row_cols)
        placements = _candidate_placements(mask_set, shape, "h", row, "left", ordered[0])
        if placements:
            candidates.append(
                Candidate(
                    axis="h",
                    line_value=row,
                    side="left",
                    segment_count=_count_segments(ordered),
                    hit_cells=tuple((row, c) for c in ordered),
                    placements=placements,
                )
            )
        placements = _candidate_placements(mask_set, shape, "h", row, "right", ordered[-1])
        if placements:
            candidates.append(
                Candidate(
                    axis="h",
                    line_value=row,
                    side="right",
                    segment_count=_count_segments(ordered),
                    hit_cells=tuple((row, c) for c in ordered),
                    placements=placements,
                )
            )

    for col, col_rows in cols.items():
        ordered = sorted(col_rows)
        placements = _candidate_placements(mask_set, shape, "v", col, "top", ordered[0])
        if placements:
            candidates.append(
                Candidate(
                    axis="v",
                    line_value=col,
                    side="top",
                    segment_count=_count_segments(ordered),
                    hit_cells=tuple((r, col) for r in ordered),
                    placements=placements,
                )
            )
        placements = _candidate_placements(mask_set, shape, "v", col, "bottom", ordered[-1])
        if placements:
            candidates.append(
                Candidate(
                    axis="v",
                    line_value=col,
                    side="bottom",
                    segment_count=_count_segments(ordered),
                    hit_cells=tuple((r, col) for r in ordered),
                    placements=placements,
                )
            )
    return candidates


def _placements_compatible(
    occupied: set[tuple[int, int]],
    cells: tuple[tuple[int, int], ...],
) -> bool:
    cell_set = set(cells)
    if occupied & cell_set:
        return False
    for r, c in cells:
        for nbr in neighbors4(r, c):
            if nbr in occupied:
                return False
    return True


def _assign_placements(
    rng: np.random.Generator,
    chosen: list[Candidate],
) -> list[Placement] | None:
    order = sorted(range(len(chosen)), key=lambda idx: len(chosen[idx].placements))
    solution: list[Placement | None] = [None] * len(chosen)
    occupied: set[tuple[int, int]] = set()

    def backtrack(depth: int) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        options = list(chosen[idx].placements)
        rng.shuffle(options)
        for placement in options:
            if not _placements_compatible(occupied, placement.cells):
                continue
            for cell in placement.cells:
                occupied.add(cell)
            solution[idx] = placement
            if backtrack(depth + 1):
                return True
            for cell in placement.cells:
                occupied.remove(cell)
            solution[idx] = None
        return False

    if not backtrack(0):
        return None
    return [placement for placement in solution if placement is not None]


def _build_input_from_specs(
    mask_set: set[tuple[int, int]],
    shape: tuple[int, int],
    colors: list[int],
    chosen: list[Candidate],
    placements: list[Placement],
) -> tuple[np.ndarray, list[Ray]]:
    grid = np.zeros(shape, dtype=int)
    for r, c in mask_set:
        grid[r, c] = GREY

    rays: list[Ray] = []
    for color, candidate, placement in zip(colors, chosen, placements, strict=True):
        for r, c in placement.cells:
            grid[r, c] = color
        rays.append(
            Ray(
                color=color,
                marker=placement.marker,
                ray_dir=placement.ray_dir,
                component_cells=placement.cells,
                hit_cells=tuple(candidate.hit_cells),
            )
        )
    return grid, rays


def _manual_expected(grid: np.ndarray, rays: list[Ray]) -> np.ndarray:
    out = grid.copy()
    for ray in sorted(rays, key=lambda item: (item.marker[0], item.marker[1])):
        for r, c in _collect_grey_hits(grid, ray.marker, ray.ray_dir):
            out[r, c] = ray.color
    out[(grid != BLACK) & (grid != GREY)] = BLACK
    return out


def _actual_hits_match(grid: np.ndarray, rays: list[Ray]) -> bool:
    for ray in rays:
        if _collect_grey_hits(grid, ray.marker, ray.ray_dir) != ray.hit_cells:
            return False
    return True


def _has_visible_order_overlap(expected: np.ndarray, rays: list[Ray]) -> bool:
    ordered = sorted(rays, key=lambda item: (item.marker[0], item.marker[1]))
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            shared = set(ordered[i].hit_cells) & set(ordered[j].hit_cells)
            if not shared or ordered[i].color == ordered[j].color:
                continue
            sr, sc = next(iter(shared))
            if int(expected[sr, sc]) == ordered[j].color:
                return True
    return False


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    local_mask = _build_mask(rng)
    if local_mask is None:
        return None
    mask_set = _embed_mask(rng, local_mask)
    if mask_set is None:
        return None

    r0, c0, r1, c1 = _bbox(mask_set)
    shape = (r1 + 4, c1 + 4)
    shape = (max(shape[0], 13), max(shape[1], 13))
    if max(shape) > 18:
        return None

    candidates = _enumerate_candidates(mask_set, shape)
    if len(candidates) < 3:
        return None

    overlap_pairs = [
        (i, j)
        for i in range(len(candidates))
        for j in range(i + 1, len(candidates))
        if candidates[i].axis != candidates[j].axis and set(candidates[i].hit_cells) & set(candidates[j].hit_cells)
    ]
    if not overlap_pairs:
        return None

    for _ in range(200):
        target_count = 4 if len(candidates) >= 4 and float(rng.random()) < 0.55 else 3
        pair_i, pair_j = overlap_pairs[int(rng.integers(0, len(overlap_pairs)))]
        chosen = [candidates[pair_i], candidates[pair_j]]
        used_lines = {(chosen[0].axis, chosen[0].line_value), (chosen[1].axis, chosen[1].line_value)}
        remaining = [
            cand
            for idx, cand in enumerate(candidates)
            if idx not in {pair_i, pair_j} and (cand.axis, cand.line_value) not in used_lines
        ]
        rng.shuffle(remaining)

        if not any(cand.segment_count > 1 for cand in chosen):
            multi = [cand for cand in remaining if cand.segment_count > 1]
            if not multi:
                continue
            picked = multi[int(rng.integers(0, len(multi)))]
            chosen.append(picked)
            used_lines.add((picked.axis, picked.line_value))
            remaining = [cand for cand in remaining if (cand.axis, cand.line_value) not in used_lines]

        while len(chosen) < target_count and remaining:
            picked = remaining.pop()
            chosen.append(picked)
            used_lines.add((picked.axis, picked.line_value))
            remaining = [cand for cand in remaining if (cand.axis, cand.line_value) not in used_lines]

        if len(chosen) < 3:
            continue
        if not any(cand.segment_count > 1 for cand in chosen):
            continue
        if not any(cand.axis == "h" for cand in chosen) or not any(cand.axis == "v" for cand in chosen):
            continue

        placements = _assign_placements(rng, chosen)
        if placements is None:
            continue

        colors = list(MARKER_COLORS)
        rng.shuffle(colors)
        colors = colors[: len(chosen)]
        grid, rays = _build_input_from_specs(mask_set, shape, colors, chosen, placements)
        if not _actual_hits_match(grid, rays):
            continue

        parsed = _identify_rays(grid)
        parsed_keys = sorted((ray.color, ray.marker, ray.ray_dir, ray.hit_cells) for ray in parsed)
        spec_keys = sorted((ray.color, ray.marker, ray.ray_dir, ray.hit_cells) for ray in rays)
        if parsed_keys != spec_keys:
            continue

        expected = _manual_expected(grid, rays)
        if not np.array_equal(main(grid), expected):
            continue
        if not _has_visible_order_overlap(expected, rays):
            continue

        mask_cells = [expected[r, c] for r, c in mask_set]
        painted_colors = sorted(set(int(v) for v in mask_cells) - {GREY})
        if len(painted_colors) < 2:
            continue
        if all(int(v) != GREY for v in mask_cells):
            continue
        if sum(int(v) != GREY for v in mask_cells) < 5:
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(600):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selective ray-mask coloring input")
