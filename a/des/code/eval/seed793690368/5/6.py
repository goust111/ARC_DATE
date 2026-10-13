from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GREY = 5
RNG = random.Random()

BG_CHOICES = [0, 1, 2, 3, 4, 6]
COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 8, 9]

SHAPE_H_RANGE = (5, 9)
SHAPE_W_RANGE = (6, 10)
SHAPE_AREA_RANGE = (14, 28)
PIECE_COUNT_CHOICES = [3, 4, 5]
MAP_GAP_RANGE = (2, 4)
MAP_COLOR_COUNT_CHOICES = [3, 4, 5]
CANVAS_MARGIN_RANGE = (5, 9)
MAX_GEN_RETRIES = 400
MAX_LOCAL_RETRIES = 80
PIECE_PLACE_RETRIES = 300


def get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _propagate_visible_colors(crop: np.ndarray, bg: int, grey: int) -> np.ndarray:
    out = np.array(crop, dtype=int, copy=True)
    h, w = out.shape
    visible = (out != bg) & (out != grey)
    unknown = ~visible
    assigned = visible.copy()
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if visible[r, c]:
                q.append((r, c))
    while q:
        r, c = q.popleft()
        color = int(out[r, c])
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and unknown[nr, nc] and not assigned[nr, nc]:
                assigned[nr, nc] = True
                out[nr, nc] = color
                q.append((nr, nc))
    if np.any(~assigned):
        raise AssertionError("color propagation left unreachable cells")
    return out


def _candidate_placements(
    hole_mask: np.ndarray, shape: tuple[tuple[int, int], ...]
) -> list[int]:
    h, w = hole_mask.shape
    ph, pw = _shape_dims(shape)
    hole_coords = [(r, c) for r in range(h) for c in range(w) if hole_mask[r, c]]
    index = {coord: i for i, coord in enumerate(hole_coords)}
    opts: set[int] = set()
    for off_r in range(h - ph + 1):
        for off_c in range(w - pw + 1):
            bitmask = 0
            ok = True
            for r, c in shape:
                rr, cc = r + off_r, c + off_c
                if not hole_mask[rr, cc]:
                    ok = False
                    break
                bitmask |= 1 << index[(rr, cc)]
            if ok:
                opts.add(bitmask)
    return sorted(opts)


def _exact_cover_solutions(
    hole_mask: np.ndarray,
    shapes: list[tuple[tuple[int, int], ...]],
    limit: int = 2,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    h, w = hole_mask.shape
    hole_coords = [(r, c) for r in range(h) for c in range(w) if hole_mask[r, c]]
    cell_count = len(hole_coords)
    all_mask = (1 << cell_count) - 1
    options = [_candidate_placements(hole_mask, shape) for shape in shapes]
    if any(not opts for opts in options):
        return [], hole_coords
    target_areas = [len(shape) for shape in shapes]
    full_used = (1 << len(shapes)) - 1
    solutions: list[list[tuple[int, int]]] = []

    def dfs(used_mask: int, covered_mask: int, placed_area: int, acc: list[tuple[int, int]]) -> None:
        if len(solutions) >= limit:
            return
        if covered_mask == all_mask:
            if used_mask == full_used and placed_area == cell_count:
                solutions.append(acc.copy())
            return
        remaining_area = sum(target_areas[i] for i in range(len(shapes)) if not (used_mask >> i) & 1)
        if placed_area + remaining_area != cell_count:
            return

        uncovered = all_mask & ~covered_mask
        best: list[tuple[int, int]] | None = None
        probe = uncovered
        while probe:
            bit = probe & -probe
            cell_idx = bit.bit_length() - 1
            cands: list[tuple[int, int]] = []
            for piece_idx, piece_opts in enumerate(options):
                if (used_mask >> piece_idx) & 1:
                    continue
                for opt in piece_opts:
                    if (opt & bit) and not (opt & covered_mask):
                        cands.append((piece_idx, opt))
            if not cands:
                return
            if best is None or len(cands) < len(best):
                best = cands
                if len(best) == 1:
                    break
            probe ^= bit

        assert best is not None
        for piece_idx, opt in best:
            acc.append((piece_idx, opt))
            dfs(
                used_mask | (1 << piece_idx),
                covered_mask | opt,
                placed_area + target_areas[piece_idx],
                acc,
            )
            acc.pop()

    dfs(0, 0, 0, [])
    return solutions, hole_coords


def _solution_to_coords(
    solution: list[tuple[int, int]], hole_coords: list[tuple[int, int]]
) -> dict[int, list[tuple[int, int]]]:
    by_piece: dict[int, list[tuple[int, int]]] = {}
    for piece_idx, opt in solution:
        coords: list[tuple[int, int]] = []
        probe = opt
        while probe:
            bit = probe & -probe
            cell_idx = bit.bit_length() - 1
            coords.append(hole_coords[cell_idx])
            probe ^= bit
        by_piece[piece_idx] = sorted(coords)
    return by_piece


def _majority_color(values: list[int]) -> int:
    counts = Counter(map(int, values))
    best = counts.most_common()
    if len(best) > 1 and best[0][1] == best[1][1]:
        raise AssertionError("piece coloring must have a unique majority")
    return int(best[0][0])


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    grey = GREY if np.any(g == GREY) else int([v for v in np.unique(g) if v != bg][0])

    color_mask = (g != bg) & (g != grey)
    if not np.any(color_mask):
        raise AssertionError("missing color map")
    map_coords = list(map(tuple, np.argwhere(color_mask)))
    mr0, mc0, mr1, mc1 = _bbox(map_coords)

    crop = g[mr0 : mr1 + 1, mc0 : mc1 + 1]
    outline_mask = crop == grey
    hole_mask = crop == bg
    if not np.any(outline_mask) or not np.any(hole_mask):
        raise AssertionError("missing outline or hole")

    full_map = _propagate_visible_colors(crop, bg, grey)

    piece_mask = g == grey
    piece_mask[mr0 : mr1 + 1, mc0 : mc1 + 1] = False
    piece_comps = _components(piece_mask, connectivity=4)
    if not piece_comps:
        raise AssertionError("missing fragments")
    shapes = [_normalize(comp) for comp in piece_comps]

    solutions, hole_coords = _exact_cover_solutions(hole_mask, shapes, limit=2)
    if len(solutions) != 1:
        raise AssertionError("fragment placement must be unique")
    placement = _solution_to_coords(solutions[0], hole_coords)

    obox = _bbox(list(map(tuple, np.argwhere(outline_mask))))
    or0, oc0, or1, oc1 = obox
    out = np.full((or1 - or0 + 1, oc1 - oc0 + 1), bg, dtype=int)
    for r, c in np.argwhere(outline_mask):
        out[int(r - or0), int(c - oc0)] = grey

    for piece_idx, coords in placement.items():
        color = _majority_color([int(full_map[r, c]) for r, c in coords])
        for r, c in coords:
            out[r - or0, c - oc0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _random_connected_mask(h: int, w: int, area: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()
    for dr, dc in DIRS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))
    while int(mask.sum()) < area:
        if not frontier:
            return None
        ranked = []
        for cell in frontier:
            rr, cc = cell
            touch = sum(
                1
                for dr, dc in DIRS4
                if 0 <= rr + dr < h and 0 <= cc + dc < w and mask[rr + dr, cc + dc]
            )
            ranked.append((touch, rng.random(), cell))
        ranked.sort(reverse=True)
        keep = max(1, min(len(ranked), 5))
        rr, cc = rng.choice([cell for _, _, cell in ranked[:keep]])
        mask[rr, cc] = True
        frontier.discard((rr, cc))
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))

    coords = list(map(tuple, np.argwhere(mask)))
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1) < 4 or (c1 - c0 + 1) < 4:
        return None
    if area == (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    return mask


def _choose_far_seeds(
    coords: list[tuple[int, int]], count: int, rng: random.Random
) -> list[tuple[int, int]]:
    seeds = [rng.choice(coords)]
    while len(seeds) < count:
        ranked = []
        for cell in coords:
            if cell in seeds:
                continue
            dist = min(abs(cell[0] - sr) + abs(cell[1] - sc) for sr, sc in seeds)
            ranked.append((dist, rng.random(), cell))
        ranked.sort(reverse=True)
        take = max(1, min(len(ranked), 6))
        seeds.append(rng.choice([cell for _, _, cell in ranked[:take]]))
    return seeds


def _partition_connected(
    mask: np.ndarray, count: int, rng: random.Random
) -> list[list[tuple[int, int]]] | None:
    coords = list(map(tuple, np.argwhere(mask)))
    if len(coords) < count:
        return None
    labels = -np.ones(mask.shape, dtype=int)
    seeds = _choose_far_seeds(coords, count, rng)
    regions: list[set[tuple[int, int]]] = []
    for idx, seed in enumerate(seeds):
        labels[seed] = idx
        regions.append({seed})
    unassigned = len(coords) - count

    while unassigned > 0:
        active: list[int] = []
        for idx, region in enumerate(regions):
            found = False
            for r, c in region:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and labels[nr, nc] < 0:
                        found = True
                        break
                if found:
                    active.append(idx)
                    break
        if not active:
            return None
        min_size = min(len(regions[idx]) for idx in active)
        growables = [idx for idx in active if len(regions[idx]) <= min_size + 1]
        idx = rng.choice(growables)
        boundary: list[tuple[int, int, int]] = []
        for r, c in regions[idx]:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and labels[nr, nc] < 0:
                    shared = sum(
                        1
                        for dr2, dc2 in DIRS4
                        if 0 <= nr + dr2 < mask.shape[0]
                        and 0 <= nc + dc2 < mask.shape[1]
                        and labels[nr + dr2, nc + dc2] == idx
                    )
                    boundary.append((shared, nr, nc))
        if not boundary:
            return None
        boundary.sort(reverse=True)
        top = boundary[: min(6, len(boundary))]
        _, nr, nc = rng.choice(top)
        labels[nr, nc] = idx
        regions[idx].add((nr, nc))
        unassigned -= 1

    parts = [sorted(region) for region in regions]
    if min(len(part) for part in parts) < 3:
        return None
    if len({_normalize(part) for part in parts}) != len(parts):
        return None
    return parts


def _grow_color_regions(
    visible_mask: np.ndarray,
    colors: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    coords = list(map(tuple, np.argwhere(visible_mask)))
    if len(coords) < len(colors):
        return None
    labels = -np.ones(visible_mask.shape, dtype=int)
    seeds = _choose_far_seeds(coords, len(colors), rng)
    regions: list[set[tuple[int, int]]] = []
    for idx, seed in enumerate(seeds):
        labels[seed] = idx
        regions.append({seed})
    unassigned = len(coords) - len(colors)

    while unassigned > 0:
        active: list[int] = []
        for idx, region in enumerate(regions):
            found = False
            for r, c in region:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < visible_mask.shape[0]
                        and 0 <= nc < visible_mask.shape[1]
                        and visible_mask[nr, nc]
                        and labels[nr, nc] < 0
                    ):
                        found = True
                        break
                if found:
                    active.append(idx)
                    break
        if not active:
            return None
        min_size = min(len(regions[idx]) for idx in active)
        expanders = [idx for idx in active if len(regions[idx]) <= min_size + 2]
        idx = rng.choice(expanders)
        candidates: list[tuple[int, int, int]] = []
        for r, c in regions[idx]:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < visible_mask.shape[0]
                    and 0 <= nc < visible_mask.shape[1]
                    and visible_mask[nr, nc]
                    and labels[nr, nc] < 0
                ):
                    shared = sum(
                        1
                        for dr2, dc2 in DIRS4
                        if 0 <= nr + dr2 < visible_mask.shape[0]
                        and 0 <= nc + dc2 < visible_mask.shape[1]
                        and labels[nr + dr2, nc + dc2] == idx
                    )
                    candidates.append((shared, nr, nc))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        top = candidates[: min(8, len(candidates))]
        _, nr, nc = rng.choice(top)
        labels[nr, nc] = idx
        regions[idx].add((nr, nc))
        unassigned -= 1

    out = np.full(visible_mask.shape, -1, dtype=int)
    for idx, color in enumerate(colors):
        for r, c in regions[idx]:
            out[r, c] = int(color)
    if min(len(region) for region in regions) < 5:
        return None
    return out


def _place_scattered_pieces(
    grid: np.ndarray,
    map_bbox: tuple[int, int, int, int],
    pieces: list[tuple[tuple[int, int], ...]],
    grey: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    mr0, mc0, mr1, mc1 = map_bbox
    occupied = np.zeros((h, w), dtype=bool)
    rr0 = max(0, mr0 - 1)
    cc0 = max(0, mc0 - 1)
    rr1 = min(h, mr1 + 2)
    cc1 = min(w, mc1 + 2)
    occupied[rr0:rr1, cc0:cc1] = True

    order = list(range(len(pieces)))
    rng.shuffle(order)
    for idx in order:
        shape = pieces[idx]
        ph, pw = _shape_dims(shape)
        placed = False
        for _ in range(PIECE_PLACE_RETRIES):
            top = rng.randrange(0, h - ph + 1)
            left = rng.randrange(0, w - pw + 1)
            pad_r0 = max(0, top - 1)
            pad_c0 = max(0, left - 1)
            pad_r1 = min(h, top + ph + 1)
            pad_c1 = min(w, left + pw + 1)
            if occupied[pad_r0:pad_r1, pad_c0:pad_c1].any():
                continue
            cells = [(top + r, left + c) for r, c in shape]
            if any(grid[r, c] != get_bg(grid) for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = grey
            occupied[pad_r0:pad_r1, pad_c0:pad_c1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _build_expected_output(
    full_map: np.ndarray,
    border_mask: np.ndarray,
    hole_parts: list[list[tuple[int, int]]],
    bg: int,
    grey: int,
) -> np.ndarray:
    obox = _bbox(list(map(tuple, np.argwhere(border_mask))))
    r0, c0, r1, c1 = obox
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in np.argwhere(border_mask):
        out[int(r - r0), int(c - c0)] = grey
    for part in hole_parts:
        color = _majority_color([int(full_map[r, c]) for r, c in part])
        for r, c in part:
            out[r - r0, c - c0] = color
    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    bg = rng.choice(BG_CHOICES)
    color_pool = [c for c in COLOR_POOL if c not in (bg, GREY)]
    rng.shuffle(color_pool)

    for _ in range(MAX_LOCAL_RETRIES):
        sh = rng.randint(*SHAPE_H_RANGE)
        sw = rng.randint(*SHAPE_W_RANGE)
        max_area = min(SHAPE_AREA_RANGE[1], sh * sw - 2)
        min_area = min(SHAPE_AREA_RANGE[0], max_area)
        if min_area < 12:
            continue
        area = rng.randint(min_area, max_area)
        hole_mask = _random_connected_mask(sh, sw, area, rng)
        if hole_mask is None:
            continue

        part_count = rng.choice([k for k in PIECE_COUNT_CHOICES if k < area // 2])
        hole_parts = _partition_connected(hole_mask, part_count, rng)
        if hole_parts is None:
            continue

        shapes = [_normalize(part) for part in hole_parts]
        solutions, _ = _exact_cover_solutions(hole_mask, shapes, limit=2)
        if len(solutions) != 1:
            continue

        top_gap = rng.randint(*MAP_GAP_RANGE)
        bottom_gap = rng.randint(*MAP_GAP_RANGE)
        left_gap = rng.randint(*MAP_GAP_RANGE)
        right_gap = rng.randint(*MAP_GAP_RANGE)
        map_h = sh + top_gap + bottom_gap
        map_w = sw + left_gap + right_gap

        hole_rect = np.zeros((map_h, map_w), dtype=bool)
        hole_rect[top_gap : top_gap + sh, left_gap : left_gap + sw] = hole_mask
        border_mask = _dilate4(hole_rect) & ~hole_rect
        if np.any(border_mask[0]) or np.any(border_mask[-1]) or np.any(border_mask[:, 0]) or np.any(border_mask[:, -1]):
            continue

        visible_mask = ~(hole_rect | border_mask)
        if not np.all(visible_mask[0]) or not np.all(visible_mask[-1]) or not np.all(visible_mask[:, 0]) or not np.all(visible_mask[:, -1]):
            continue

        color_count = rng.choice(MAP_COLOR_COUNT_CHOICES)
        colors = color_pool[:color_count]
        visible_colors = _grow_color_regions(visible_mask, colors, rng)
        if visible_colors is None:
            continue

        map_crop = np.full((map_h, map_w), bg, dtype=int)
        map_crop[visible_mask] = visible_colors[visible_mask]
        full_map = _propagate_visible_colors(map_crop, bg, GREY)

        local_parts = [
            [(r + top_gap, c + left_gap) for r, c in part]
            for part in hole_parts
        ]
        piece_color_sets = [set(int(full_map[r, c]) for r, c in part) for part in local_parts]
        if not any(len(s) >= 2 for s in piece_color_sets):
            continue
        try:
            for part in local_parts:
                _majority_color([int(full_map[r, c]) for r, c in part])
        except AssertionError:
            continue

        out = _build_expected_output(full_map, border_mask, local_parts, bg, GREY)

        top_margin = rng.randint(*CANVAS_MARGIN_RANGE)
        bottom_margin = rng.randint(*CANVAS_MARGIN_RANGE)
        left_margin = rng.randint(*CANVAS_MARGIN_RANGE)
        right_margin = rng.randint(*CANVAS_MARGIN_RANGE)
        canvas_h = map_h + top_margin + bottom_margin
        canvas_w = map_w + left_margin + right_margin
        grid = np.full((canvas_h, canvas_w), bg, dtype=int)
        map_top = top_margin
        map_left = left_margin
        grid[map_top : map_top + map_h, map_left : map_left + map_w] = full_map
        grid[map_top : map_top + map_h, map_left : map_left + map_w][border_mask] = GREY
        grid[map_top : map_top + map_h, map_left : map_left + map_w][hole_rect] = bg

        global_shapes = shapes[:]
        ok = _place_scattered_pieces(
            grid,
            (map_top, map_left, map_top + map_h - 1, map_left + map_w - 1),
            global_shapes,
            GREY,
            rng,
        )
        if not ok:
            continue

        if get_bg(grid) != bg:
            continue
        if len(_components(grid == GREY, connectivity=4)) != len(shapes):
            # Outline can be diagonally connected only; count external fragments by masking out the map rectangle.
            piece_mask = grid == GREY
            piece_mask[map_top : map_top + map_h, map_left : map_left + map_w] = False
            if len(_components(piece_mask, connectivity=4)) != len(shapes):
                continue

        solved = _solve(grid)
        if not np.array_equal(solved, out):
            continue
        return grid, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        built = _gen_attempt(RNG)
        if built is not None:
            grid, _ = built
            return grid
    raise RuntimeError("failed to generate a valid puzzle")
