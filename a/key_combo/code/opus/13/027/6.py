from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
Box = tuple[int, int, int, int]  # r0, c0, h, w

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEP2 = [(-2, 0), (2, 0), (0, -2), (0, 2)]
NON_BG = tuple(range(1, 10))
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_cells(box: Box) -> set[Coord]:
    r0, c0, h, w = box
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def box_adjacent_sites(box: Box) -> set[Coord]:
    r0, c0, h, w = box
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    out = set()
    for r in range(r0, r1 + 1):
        out.add((r, c0 - 1))
        out.add((r, c1 + 1))
    for c in range(c0, c1 + 1):
        out.add((r0 - 1, c))
        out.add((r1 + 1, c))
    return out


def box_neighbors4(box: Box, shape: tuple[int, int]) -> set[Coord]:
    return {p for p in box_adjacent_sites(box) if in_bounds(shape, p[0], p[1])}


def touches_box(site: Coord, box: Box) -> bool:
    r, c = site
    r0, c0, h, w = box
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    return ((r0 <= r <= r1) and (c in {c0 - 1, c1 + 1})) or ((c0 <= c <= c1) and (r in {r0 - 1, r1 + 1}))


def mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return True
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def region_base_and_mask(sub: np.ndarray) -> tuple[int, np.ndarray]:
    colors, counts = np.unique(sub, return_counts=True)
    assert len(colors) == 2
    base = int(colors[np.argmax(counts)])
    mask = sub != base
    return base, mask


def is_checker_window(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (3, 3) or np.any(sub == bg):
        return False
    vals = np.unique(sub)
    if len(vals) != 2:
        return False
    even = int(sub[0, 0])
    odd = int(sub[0, 1])
    if even == odd:
        return False
    for r in range(3):
        for c in range(3):
            need = even if (r + c) % 2 == 0 else odd
            if int(sub[r, c]) != need:
                return False
    return True


def _rect_can_extend(grid: np.ndarray, bg: int, r0: int, c0: int, h: int, w: int) -> bool:
    H, W = grid.shape
    if r0 > 0 and np.all(grid[r0 - 1, c0 : c0 + w] != bg):
        return True
    if r0 + h < H and np.all(grid[r0 + h, c0 : c0 + w] != bg):
        return True
    if c0 > 0 and np.all(grid[r0 : r0 + h, c0 - 1] != bg):
        return True
    if c0 + w < W and np.all(grid[r0 : r0 + h, c0 + w] != bg):
        return True
    return False


def find_regions(grid: np.ndarray, bg: int) -> list[Box]:
    H, W = grid.shape
    candidates: list[Box] = []
    for h in (5, 4):
        for w in (5, 4):
            for r0 in range(H - h + 1):
                for c0 in range(W - w + 1):
                    sub = grid[r0 : r0 + h, c0 : c0 + w]
                    if np.any(sub == bg):
                        continue
                    vals, counts = np.unique(sub, return_counts=True)
                    if len(vals) != 2:
                        continue
                    minor = int(sub.size - counts.max())
                    if minor < 2 or minor > 4:
                        continue
                    base, motif_mask = region_base_and_mask(sub)
                    if not mask_connected(motif_mask):
                        continue
                    if _rect_can_extend(grid, bg, r0, c0, h, w):
                        continue
                    candidates.append((r0, c0, h, w))
    candidates.sort(key=lambda box: box[2] * box[3], reverse=True)
    chosen: list[Box] = []
    occupied: set[Coord] = set()
    for box in candidates:
        cells = bbox_cells(box)
        if cells & occupied:
            continue
        chosen.append(box)
        occupied |= cells
    return chosen


def find_checker_candidates(grid: np.ndarray, bg: int) -> list[Box]:
    H, W = grid.shape
    out = []
    for r0 in range(H - 2):
        for c0 in range(W - 2):
            if is_checker_window(grid[r0 : r0 + 3, c0 : c0 + 3], bg):
                out.append((r0, c0, 3, 3))
    return out


def trace_guide_component(start: Coord, guide_nodes: set[Coord]) -> set[Coord]:
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in STEP2:
            nxt = (r + dr, c + dc)
            if nxt in guide_nodes and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def analyze_grid(grid: np.ndarray) -> tuple[list[Box], Box, Coord, int, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    regions = find_regions(g, bg)
    assert len(regions) >= 3, "expected 3-4 rectangular motif regions"

    checker = None
    start = None
    path_color = None
    region_cells = set().union(*(bbox_cells(box) for box in regions))
    for cand in find_checker_candidates(g, bg):
        obj_cells = region_cells | bbox_cells(cand)
        adj = []
        for rr, cc in box_neighbors4(cand, g.shape):
            if (rr, cc) in obj_cells or int(g[rr, cc]) == bg:
                continue
            adj.append((rr, cc, int(g[rr, cc])))
        if len(adj) == 1:
            checker = cand
            start = (adj[0][0], adj[0][1])
            path_color = adj[0][2]
            break
    assert checker is not None and start is not None and path_color is not None, "checkerboard source not found"

    obj_mask = np.zeros(g.shape, dtype=bool)
    for box in regions + [checker]:
        r0, c0, h, w = box
        obj_mask[r0 : r0 + h, c0 : c0 + w] = True
    guide_nodes = {
        (int(r), int(c))
        for r, c in np.argwhere((g == path_color) & (~obj_mask))
    }
    assert start in guide_nodes, "start guide pixel missing"

    first = trace_guide_component(start, guide_nodes)
    x_hits = [i for i, box in enumerate(regions) if any(touches_box(site, box) for site in first)]
    assert len(x_hits) == 1, "first guide segment must point to exactly one region"
    x_idx = x_hits[0]

    next_starts = {site for site in guide_nodes - first if touches_box(site, regions[x_idx])}
    assert next_starts, "second guide segment must leave the first region"
    second = set()
    for site in next_starts:
        if site not in second:
            second |= trace_guide_component(site, guide_nodes)
    y_hits = [i for i, box in enumerate(regions) if i != x_idx and any(touches_box(site, box) for site in second)]
    assert len(y_hits) == 1, "second guide segment must point to exactly one other region"
    y_idx = y_hits[0]
    return regions, checker, start, path_color, x_idx, y_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap the motif masks of the two regions selected by the checker-linked guide path.

    Input structure:
    - A mostly black grid contains one 3x3 checkerboard source, 3-4 filled rectangular regions, and sparse guide pixels placed on a two-cell lattice.
    - Each rectangular region is a 4x4 to 5x5 box made from one dominant base color plus a small 4-connected minority-color motif.
    - The checkerboard has one adjacent guide pixel; that guide color also appears in a second disconnected segment leaving the first reached region toward a second region.

    Rule:
    - Find the checkerboard and read the color of its unique adjacent guide pixel.
    - Follow same-color guide pixels in horizontal or vertical jumps of two cells to the first adjacent rectangular region X.
    - From guide pixels of the same color on the opposite side of X, follow the second segment to the other adjacent rectangular region Y.
    - Clear both regions to their own base colors, then paste Y's motif mask into X using Y's base color and paste X's motif mask into Y using X's base color.

    Key insight:
    - Only the guide color attached to the checkerboard matters; the path is split into two disconnected guide segments bridged logically by the first reached region, while other guide paths are distractors.
    """
    g = np.asarray(grid, dtype=int)
    regions, _checker, _start, _path_color, x_idx, y_idx = analyze_grid(g)

    x_box = regions[x_idx]
    y_box = regions[y_idx]
    xr, xc, xh, xw = x_box
    yr, yc, yh, yw = y_box
    x_sub = g[xr : xr + xh, xc : xc + xw]
    y_sub = g[yr : yr + yh, yc : yc + yw]
    x_base, x_mask = region_base_and_mask(x_sub)
    y_base, y_mask = region_base_and_mask(y_sub)

    out = g.copy()
    out[xr : xr + xh, xc : xc + xw] = x_base
    out[yr : yr + yh, yc : yc + yw] = y_base
    out[xr : xr + xh, xc : xc + xw][y_mask] = y_base
    out[yr : yr + yh, yc : yc + yw][x_mask] = x_base
    return out


def random_motif_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    target = int(rng.integers(2, min(4, len(inner)) + 1))
    start = inner[int(rng.integers(len(inner)))]
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in N4:
                nr, nc = r + dr, c + dc
                if 1 <= nr < h - 1 and 1 <= nc < w - 1 and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(frontier[int(rng.integers(len(frontier)))])
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def box_far_apart(a: Box, b: Box, gap: int = 2) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1, ac1 = ar0 + ah - 1, ac0 + aw - 1
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def place_checker(shape: tuple[int, int], regions: list[Box], rng: np.random.Generator) -> Box | None:
    H, W = shape
    row_lo = max(1, H // 2 - 3)
    row_hi = min(H - 4, H // 2 + 1)
    col_lo = max(1, W // 2 - 3)
    col_hi = min(W - 4, W // 2 + 1)
    tries = []
    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(col_lo, col_hi + 1):
            box = (r0, c0, 3, 3)
            if all(box_far_apart(box, reg, gap=2) for reg in regions):
                tries.append(box)
    if not tries:
        return None
    return tries[int(rng.integers(len(tries)))]


def place_regions(shape: tuple[int, int], region_count: int, rng: np.random.Generator) -> list[Box] | None:
    H, W = shape
    mid_r = H // 2
    mid_c = W // 2
    slots = {
        "TL": ((1, max(1, mid_r - 6)), (1, max(1, mid_c - 6))),
        "TR": ((1, max(1, mid_r - 6)), (min(W - 6, mid_c + 1), W - 6)),
        "BL": ((min(H - 6, mid_r + 1), H - 6), (1, max(1, mid_c - 6))),
        "BR": ((min(H - 6, mid_r + 1), H - 6), (min(W - 6, mid_c + 1), W - 6)),
    }
    chosen_slots = list(rng.choice(list(slots.keys()), size=region_count, replace=False))
    out = []
    for name in chosen_slots:
        (rlo, rhi), (clo, chi) = slots[name]
        if rlo > rhi or clo > chi:
            return None
        options = []
        for h in (4, 5):
            for w in (4, 5):
                for r0 in range(rlo, rhi + 1):
                    for c0 in range(clo, chi + 1):
                        if r0 + h <= H - 1 and c0 + w <= W - 1:
                            box = (r0, c0, h, w)
                            if all(box_far_apart(box, prev, gap=2) for prev in out):
                                options.append(box)
        if not options:
            return None
        out.append(options[int(rng.integers(len(options)))])
    return out


def attachment_candidates(box: Box, parity: tuple[int, int], shape: tuple[int, int]) -> list[tuple[Coord, str]]:
    r0, c0, h, w = box
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    pr, pc = parity
    out = []
    for r in range(r0, r1 + 1):
        for c, side in ((c0 - 1, "L"), (c1 + 1, "R")):
            if in_bounds(shape, r, c) and r % 2 == pr and c % 2 == pc:
                out.append(((r, c), side))
    for c in range(c0, c1 + 1):
        for r, side in ((r0 - 1, "T"), (r1 + 1, "B")):
            if in_bounds(shape, r, c) and r % 2 == pr and c % 2 == pc:
                out.append(((r, c), side))
    return out


def object_adjacent_nodes(objects: Iterable[Box], shape: tuple[int, int]) -> set[Coord]:
    out = set()
    for box in objects:
        out |= box_neighbors4(box, shape)
    return out


def bfs_path(
    start: Coord,
    end: Coord,
    shape: tuple[int, int],
    blocked_nodes: set[Coord],
    avoid_nodes: set[Coord],
    special_nodes: set[Coord],
    object_adj: set[Coord],
    rng: np.random.Generator,
) -> list[Coord] | None:
    if start == end:
        return [start]
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        deltas = STEP2.copy()
        rng.shuffle(deltas)
        for dr, dc in deltas:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if nxt in parent or not in_bounds(shape, nr, nc):
                continue
            if nxt in blocked_nodes:
                continue
            if nxt in avoid_nodes and nxt not in special_nodes:
                continue
            if nxt in object_adj and nxt not in special_nodes:
                continue
            if (nr % 2, nc % 2) != (start[0] % 2, start[1] % 2):
                continue
            parent[nxt] = cur
            if nxt == end:
                path = [end]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            q.append(nxt)
    return None


def expand_same_color_separation(nodes: Iterable[Coord], shape: tuple[int, int]) -> set[Coord]:
    out = set(nodes)
    for r, c in list(nodes):
        for dr, dc in STEP2:
            nr, nc = r + dr, c + dc
            if in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def choose_path_colors(
    region_colors: list[tuple[int, int]],
    path_specs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[int] | None:
    options = []
    for a, b in path_specs:
        banned = set(region_colors[a]) | set(region_colors[b])
        choices = [c for c in NON_BG if c not in banned]
        rng.shuffle(choices)
        options.append(choices)
    order = sorted(range(len(path_specs)), key=lambda i: len(options[i]))
    assigned = [None] * len(path_specs)

    def backtrack(k: int, used: set[int]) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        for color in options[idx]:
            if color in used:
                continue
            assigned[idx] = color
            used.add(color)
            if backtrack(k + 1, used):
                return True
            used.remove(color)
        return False

    if not backtrack(0, set()):
        return None
    return [int(x) for x in assigned]


def plan_main_path(
    shape: tuple[int, int],
    checker: Box,
    regions: list[Box],
    x_idx: int,
    y_idx: int,
    blocked_nodes: set[Coord],
    object_adj: set[Coord],
    rng: np.random.Generator,
) -> tuple[list[Coord], list[Coord]] | None:
    start_sites = list(box_neighbors4(checker, shape))
    rng.shuffle(start_sites)
    for start in start_sites:
        parity = (start[0] % 2, start[1] % 2)
        x_sites = attachment_candidates(regions[x_idx], parity, shape)
        y_sites = attachment_candidates(regions[y_idx], parity, shape)
        rng.shuffle(x_sites)
        rng.shuffle(y_sites)
        for i, (x_enter, side_a) in enumerate(x_sites):
            for x_leave, side_b in x_sites[i + 1 :]:
                if side_a == side_b:
                    continue
                if abs(x_enter[0] - x_leave[0]) + abs(x_enter[1] - x_leave[1]) == 2:
                    continue
                path1 = bfs_path(
                    start,
                    x_enter,
                    shape,
                    blocked_nodes,
                    avoid_nodes=set(),
                    special_nodes={start, x_enter},
                    object_adj=object_adj,
                    rng=rng,
                )
                if path1 is None:
                    continue
                sep = expand_same_color_separation(path1, shape)
                for y_enter, _side_y in y_sites:
                    if y_enter in sep:
                        continue
                    path2 = bfs_path(
                        x_leave,
                        y_enter,
                        shape,
                        blocked_nodes,
                        avoid_nodes=sep,
                        special_nodes={x_leave, y_enter},
                        object_adj=object_adj,
                        rng=rng,
                    )
                    if path2 is not None:
                        return path1, path2
    return None


def plan_direct_path(
    shape: tuple[int, int],
    a_box: Box,
    b_box: Box,
    blocked_nodes: set[Coord],
    object_adj: set[Coord],
    avoid_nodes: set[Coord],
    rng: np.random.Generator,
) -> list[Coord] | None:
    parities = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(parities)
    for parity in parities:
        a_sites = attachment_candidates(a_box, parity, shape)
        b_sites = attachment_candidates(b_box, parity, shape)
        rng.shuffle(a_sites)
        rng.shuffle(b_sites)
        for a_site, _ in a_sites:
            for b_site, _ in b_sites:
                path = bfs_path(
                    a_site,
                    b_site,
                    shape,
                    blocked_nodes,
                    avoid_nodes=avoid_nodes,
                    special_nodes={a_site, b_site},
                    object_adj=object_adj,
                    rng=rng,
                )
                if path is not None:
                    return path
    return None


def fill_checker(grid: np.ndarray, box: Box, colors: tuple[int, int]) -> None:
    r0, c0, _h, _w = box
    a, b = colors
    for r in range(3):
        for c in range(3):
            grid[r0 + r, c0 + c] = a if (r + c) % 2 == 0 else b


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        H = int(rng.integers(19, 27))
        W = int(rng.integers(19, 27))
        region_count = 4 if rng.random() < 0.65 else 3
        regions = place_regions((H, W), region_count, rng)
        if regions is None:
            continue
        checker = place_checker((H, W), regions, rng)
        if checker is None:
            continue

        main_pair = list(rng.choice(region_count, size=2, replace=False))
        leftover = [i for i in range(region_count) if i not in main_pair]
        distractor_count = 0
        if len(leftover) >= 2:
            distractor_count = 2 if rng.random() < 0.4 else 1

        region_color_pool = list(NON_BG)
        rng.shuffle(region_color_pool)
        base_colors = region_color_pool[:region_count]
        motif_colors = region_color_pool[region_count : 2 * region_count]
        region_colors = list(zip(base_colors, motif_colors))
        checker_colors = tuple(map(int, rng.choice(NON_BG, size=2, replace=False)))

        path_specs = [(main_pair[0], main_pair[1])]
        for _ in range(distractor_count):
            path_specs.append((leftover[0], leftover[1]))
        path_colors = choose_path_colors(region_colors, path_specs, rng)
        if path_colors is None:
            continue

        grid = np.zeros((H, W), dtype=int)
        region_masks: list[np.ndarray] = []
        for i, box in enumerate(regions):
            r0, c0, h, w = box
            base, motif = region_colors[i]
            mask = random_motif_mask(h, w, rng)
            region_masks.append(mask)
            grid[r0 : r0 + h, c0 : c0 + w] = base
            grid[r0 : r0 + h, c0 : c0 + w][mask] = motif
        fill_checker(grid, checker, checker_colors)

        objects = regions + [checker]
        blocked_nodes = set().union(*(bbox_cells(box) for box in objects))
        obj_adj = object_adjacent_nodes(objects, (H, W))

        main_paths = plan_main_path((H, W), checker, regions, main_pair[0], main_pair[1], blocked_nodes, obj_adj, rng)
        if main_paths is None:
            continue
        path1, path2 = main_paths
        used_nodes = set(path1) | set(path2)
        for site in used_nodes:
            grid[site] = path_colors[0]

        ok = True
        distractor_paths: list[list[Coord]] = []
        for idx in range(distractor_count):
            avoid = used_nodes | set().union(*distractor_paths) if distractor_paths else used_nodes
            path = plan_direct_path((H, W), regions[leftover[0]], regions[leftover[1]], blocked_nodes, obj_adj, avoid, rng)
            if path is None:
                if idx == 0:
                    ok = False
                break
            distractor_paths.append(path)
            for site in path:
                grid[site] = path_colors[idx + 1]
            used_nodes |= set(path)
        if not ok:
            continue

        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        try:
            analyzed = analyze_grid(grid)
        except Exception:
            continue
        if {analyzed[4], analyzed[5]} != set(main_pair):
            continue
        return grid
    raise RuntimeError("failed to generate valid input for puzzle 6")
