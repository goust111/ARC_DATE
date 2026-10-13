from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 1
YELLOW = 4
PALETTE = [2, 3, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MAX_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _scan_nonblue_objects(
    grid: np.ndarray,
) -> tuple[list[dict[str, int | tuple[int, int, int, int]]], list[dict[str, int | tuple[int, int]]]]:
    sources = []
    seeds = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in {BG, WALL, YELLOW}:
            continue
        for comp in _components_of_color(grid, color):
            if len(comp) == 1:
                seeds.append({"color": color, "pos": comp[0]})
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if len(comp) == 4 and r1 - r0 == 1 and c1 - c0 == 1:
                if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                    sources.append({"color": color, "bbox": (r0, c0, r1, c1)})
                    continue
            raise AssertionError("non-blue foreground objects must be lone seeds or solid 2x2 sources")
    return sources, seeds


def _classify_objects(
    grid: np.ndarray,
) -> tuple[list[dict[str, int | tuple[int, int, int, int]]], list[dict[str, int | tuple[int, int]]]]:
    sources, seeds = _scan_nonblue_objects(grid)
    assert 2 <= len(sources) <= 3
    assert 2 <= len(seeds) <= 3
    return sources, seeds


def _ray_masks(
    grid: np.ndarray,
    sources: list[dict[str, int | tuple[int, int, int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    path_mask = np.zeros((h, w), dtype=bool)
    paint_mask = np.zeros((h, w), dtype=bool)

    def cast(r: int, c: int, dr: int, dc: int) -> None:
        while 0 <= r < h and 0 <= c < w:
            if int(grid[r, c]) == WALL:
                break
            path_mask[r, c] = True
            if int(grid[r, c]) == BG:
                paint_mask[r, c] = True
            r += dr
            c += dc

    for source in sources:
        r0, c0, r1, c1 = source["bbox"]  # type: ignore[index]
        for r in range(r0, r1 + 1):
            cast(r, c0 - 1, 0, -1)
            cast(r, c1 + 1, 0, 1)
        for c in range(c0, c1 + 1):
            cast(r0 - 1, c, -1, 0)
            cast(r1 + 1, c, 1, 0)

    return path_mask, paint_mask


def _nearest_corner_info(shape: tuple[int, int], pos: tuple[int, int]) -> tuple[tuple[int, int], int]:
    h, w = shape
    r, c = pos
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    dists = [abs(r - cr) + abs(c - cc) for cr, cc in corners]
    best = min(dists)
    assert dists.count(best) == 1, "seed must have a unique nearest corner"
    return corners[dists.index(best)], best


def _seed_infos(
    shape: tuple[int, int],
    seeds: list[dict[str, int | tuple[int, int]]],
) -> list[dict[str, int | tuple[int, int] | tuple[int, int, int, int]]]:
    infos = []
    for seed in seeds:
        pos = seed["pos"]  # type: ignore[index]
        corner, dist = _nearest_corner_info(shape, pos)
        r0, r1 = sorted((pos[0], corner[0]))
        c0, c1 = sorted((pos[1], corner[1]))
        infos.append(
            {
                "color": seed["color"],
                "pos": pos,
                "corner": corner,
                "distance": dist,
                "rect": (r0, c0, r1, c1),
            }
        )
    return infos


def _apply_fill(
    base: np.ndarray,
    seed_infos: list[dict[str, int | tuple[int, int] | tuple[int, int, int, int]]],
) -> np.ndarray:
    out = base.copy()
    priority = np.full(base.shape, 10**9, dtype=int)
    chosen = np.zeros(base.shape, dtype=int)
    owner = np.full(base.shape, -1, dtype=int)

    for idx, info in enumerate(seed_infos):
        dist = int(info["distance"])
        color = int(info["color"])
        r0, c0, r1, c1 = info["rect"]  # type: ignore[index]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(base[r, c]) != BG:
                    continue
                if dist < priority[r, c]:
                    priority[r, c] = dist
                    chosen[r, c] = color
                    owner[r, c] = idx
                elif dist == priority[r, c] and owner[r, c] != idx:
                    raise AssertionError("overlapping fills must not tie on nearest-corner distance")

    mask = owner >= 0
    out[mask] = chosen[mask]
    return out


def _solve_from_objects(
    grid: np.ndarray,
    sources: list[dict[str, int | tuple[int, int, int, int]]],
    seeds: list[dict[str, int | tuple[int, int]]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    _path_mask, paint_mask = _ray_masks(out, sources)
    out[paint_mask] = YELLOW
    seed_infos = _seed_infos(out.shape, seeds)
    return _apply_fill(out, seed_infos)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast blocker-stopped yellow rays from 2x2 sources, then fill each lone seed toward its nearest corner without crossing yellow.

    Input structure:
    - A black background contains 2-3 non-blue solid 2x2 rectangles, which are the ray sources.
    - Blue connected objects are walls, and 2-3 other non-blue colors appear as isolated single-pixel fill seeds.
    - Objects are separated in the input; yellow is absent and only appears in the output.

    Rule:
    - From every source row cast left/right rays, and from every source column cast up/down rays.
    - A ray stops before blue; every black cell it traverses becomes yellow, while sources, seeds, and walls stay unchanged.
    - For each isolated seed, choose its unique nearest grid corner by Manhattan distance and take the inclusive axis-aligned rectangle between the seed and that corner.
    - Fill only the black cells of that target rectangle with the seed color; yellow walls are never overwritten.
    - If multiple seeds want the same black cell, the seed with the smaller nearest-corner distance claims it.

    Key insight:
    The ray system and the corner-fill system are defined separately, but the yellow rays must be drawn first because they become visible barriers that cut the fill rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert YELLOW not in set(int(v) for v in np.unique(g))

    sources, seeds = _classify_objects(g)
    return _solve_from_objects(g, sources, seeds)


def _shape_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _can_place(cells: set[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not _in_bounds((h, w), r, c) or occupied[r, c]:
            return False
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if _in_bounds((h, w), nr, nc) and occupied[nr, nc] and (nr, nc) not in cells:
                return False
    return True


def _paint_cells(grid: np.ndarray, occupied: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color
        occupied[r, c] = True


def _grow_irregular_shape(
    area: int,
    anchor: tuple[int, int],
    shape: tuple[int, int],
    occupied: np.ndarray,
) -> set[tuple[int, int]] | None:
    for _ in range(80):
        cells = {anchor}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if not _in_bounds(shape, nr, nc) or (nr, nc) in cells:
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area or not _can_place(cells, occupied):
            continue
        r0, c0, r1, c1 = _bbox(list(cells))
        if (r1 - r0 + 1) * (c1 - c0 + 1) == area:
            continue
        return cells
    return None


def _place_sources(
    grid: np.ndarray,
    occupied: np.ndarray,
    colors: list[int],
) -> list[dict[str, int | tuple[int, int, int, int]]]:
    h, w = grid.shape
    sources = []
    for color in colors:
        placed = False
        for _ in range(120):
            r0 = random.randint(2, h - 4)
            c0 = random.randint(2, w - 4)
            cells = _shape_cells(r0, c0, 2, 2)
            if not _can_place(cells, occupied):
                continue
            _paint_cells(grid, occupied, cells, color)
            sources.append({"color": color, "bbox": (r0, c0, r0 + 1, c0 + 1)})
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place source rectangle")
    return sources


def _blocking_anchor_candidates(
    source: dict[str, int | tuple[int, int, int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = source["bbox"]  # type: ignore[index]
    anchors = []
    for r in range(r0, r1 + 1):
        for c in range(0, c0 - 1):
            anchors.append((r, c))
        for c in range(c1 + 2, w):
            anchors.append((r, c))
    for c in range(c0, c1 + 1):
        for r in range(0, r0 - 1):
            anchors.append((r, c))
        for r in range(r1 + 2, h):
            anchors.append((r, c))
    return anchors


def _place_walls(
    grid: np.ndarray,
    occupied: np.ndarray,
    sources: list[dict[str, int | tuple[int, int, int, int]]],
) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    walls: list[set[tuple[int, int]]] = []

    anchor_pool = []
    for source in sources:
        for anchor in _blocking_anchor_candidates(source, (h, w)):
            anchor_pool.append((source, anchor))
    random.shuffle(anchor_pool)

    first_wall = None
    for _source, anchor in anchor_pool:
        area = random.randint(3, 6)
        first_wall = _grow_irregular_shape(area, anchor, (h, w), occupied)
        if first_wall is not None:
            break
    if first_wall is None:
        raise RuntimeError("failed to place a blocking wall")
    _paint_cells(grid, occupied, first_wall, WALL)
    walls.append(first_wall)

    extra_count = random.randint(1, 3)
    for _ in range(extra_count):
        placed = False
        for _ in range(120):
            anchor = (random.randint(1, h - 2), random.randint(1, w - 2))
            area = random.randint(3, 6)
            wall = _grow_irregular_shape(area, anchor, (h, w), occupied)
            if wall is None:
                continue
            _paint_cells(grid, occupied, wall, WALL)
            walls.append(wall)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place all walls")
    return walls


def _candidate_seed_positions(
    grid_without_seeds: np.ndarray,
    occupied: np.ndarray,
    source_count: int,
) -> list[dict[str, int | tuple[int, int] | tuple[int, int, int, int] | np.ndarray]]:
    del source_count
    sources, _ = _scan_nonblue_objects(grid_without_seeds)
    assert 2 <= len(sources) <= 3
    path_mask, paint_mask = _ray_masks(grid_without_seeds, sources)
    ray_grid = grid_without_seeds.copy()
    ray_grid[paint_mask] = YELLOW

    h, w = grid_without_seeds.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if occupied[r, c]:
                continue
            blocked = False
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if _in_bounds((h, w), nr, nc) and occupied[nr, nc]:
                    blocked = True
                    break
            if blocked:
                continue
            try:
                corner, dist = _nearest_corner_info((h, w), (r, c))
            except AssertionError:
                continue
            r0, r1 = sorted((r, corner[0]))
            c0, c1 = sorted((c, corner[1]))
            rect = (r0, c0, r1, c1)
            black_count = int(np.count_nonzero(ray_grid[r0 : r1 + 1, c0 : c1 + 1] == BG))
            if black_count < 2:
                continue
            interact_count = int(np.count_nonzero(paint_mask[r0 : r1 + 1, c0 : c1 + 1]))
            if paint_mask[r, c]:
                interact_count -= 1
            candidates.append(
                {
                    "pos": (r, c),
                    "corner": corner,
                    "distance": dist,
                    "rect": rect,
                    "interacts": interact_count > 0,
                    "path_mask": path_mask,
                    "paint_mask": paint_mask,
                    "ray_grid": ray_grid,
                }
            )
    return candidates


def _rect_overlap_black(
    ray_grid: np.ndarray,
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
) -> int:
    r0 = max(rect_a[0], rect_b[0])
    c0 = max(rect_a[1], rect_b[1])
    r1 = min(rect_a[2], rect_b[2])
    c1 = min(rect_a[3], rect_b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int(np.count_nonzero(ray_grid[r0 : r1 + 1, c0 : c1 + 1] == BG))


def _pick_seeds(
    grid: np.ndarray,
    occupied: np.ndarray,
    seed_colors: list[int],
    overlap_required: bool,
) -> list[dict[str, int | tuple[int, int]]]:
    candidates = _candidate_seed_positions(grid, occupied, len(seed_colors))
    interaction_candidates = [cand for cand in candidates if cand["interacts"]]
    if not interaction_candidates:
        raise RuntimeError("need at least one interaction seed")

    first = random.choice(interaction_candidates)
    chosen = [first]
    used_positions = {first["pos"]}
    used_dists = {first["distance"]}
    ray_grid = first["ray_grid"]  # type: ignore[assignment]

    if len(seed_colors) >= 2 and overlap_required:
        overlapping = [
            cand
            for cand in candidates
            if cand["pos"] not in used_positions
            and cand["distance"] not in used_dists
            and _rect_overlap_black(ray_grid, first["rect"], cand["rect"]) > 0
        ]
        if not overlapping:
            raise RuntimeError("failed to find an overlapping seed pair")
        second = random.choice(overlapping)
        chosen.append(second)
        used_positions.add(second["pos"])
        used_dists.add(second["distance"])

    while len(chosen) < len(seed_colors):
        pool = [
            cand
            for cand in candidates
            if cand["pos"] not in used_positions and cand["distance"] not in used_dists
        ]
        if overlap_required and len(chosen) == 1:
            pool = [
                cand
                for cand in pool
                if _rect_overlap_black(ray_grid, first["rect"], cand["rect"]) > 0
            ]
        if not pool:
            raise RuntimeError("failed to complete seed placement")
        cand = random.choice(pool)
        chosen.append(cand)
        used_positions.add(cand["pos"])
        used_dists.add(cand["distance"])

    seeds = []
    for color, cand in zip(seed_colors, chosen):
        r, c = cand["pos"]  # type: ignore[index]
        grid[r, c] = color
        occupied[r, c] = True
        seeds.append({"color": color, "pos": (r, c)})
    return seeds


def _full_cross_without_walls(shape: tuple[int, int], sources: list[dict[str, int | tuple[int, int, int, int]]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for source in sources:
        r0, c0, r1, c1 = source["bbox"]  # type: ignore[index]
        for r in range(r0, r1 + 1):
            mask[r, :] = True
            mask[r, c0 : c1 + 1] = False
        for c in range(c0, c1 + 1):
            mask[:, c] = True
            mask[r0 : r1 + 1, c] = False
    return mask


def _validate_generated(
    grid: np.ndarray,
    overlap_required: bool,
) -> bool:
    try:
        sources, seeds = _classify_objects(grid)
    except AssertionError:
        return False

    source_colors = {int(src["color"]) for src in sources}
    seed_colors = {int(seed["color"]) for seed in seeds}
    if source_colors & seed_colors:
        return False

    path_mask, paint_mask = _ray_masks(grid, sources)
    if int(np.count_nonzero(paint_mask)) < 6:
        return False

    no_wall_cross = _full_cross_without_walls(grid.shape, sources)
    if int(np.count_nonzero(no_wall_cross & (grid == WALL))) == 0:
        return False

    ray_stage = grid.copy()
    ray_stage[paint_mask] = YELLOW
    seed_infos = _seed_infos(grid.shape, seeds)
    distances = [int(info["distance"]) for info in seed_infos]
    if len(distances) != len(set(distances)):
        return False

    interaction_count = 0
    fill_sizes = []
    overlap_visible = False
    for i, info in enumerate(seed_infos):
        r0, c0, r1, c1 = info["rect"]  # type: ignore[index]
        target = ray_stage[r0 : r1 + 1, c0 : c1 + 1]
        if np.count_nonzero(target == YELLOW) > 0:
            interaction_count += 1
        fill_sizes.append(int(np.count_nonzero(target == BG)) - 1)
        for j in range(i + 1, len(seed_infos)):
            if _rect_overlap_black(ray_stage, info["rect"], seed_infos[j]["rect"]) > 0:
                overlap_visible = True
    if interaction_count == 0:
        return False
    if any(size <= 0 for size in fill_sizes):
        return False
    if overlap_required and not overlap_visible:
        return False

    out = _solve_from_objects(grid, sources, seeds)
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False

    yellow_added = int(np.count_nonzero((out == YELLOW) & (grid == BG)))
    if yellow_added < 4:
        return False

    recolored = 0
    for seed in seeds:
        recolored += int(np.count_nonzero((out == int(seed["color"])) & (grid == BG)))
    if recolored < len(seeds):
        return False

    total = grid.size
    if yellow_added > total // 2 or recolored > total // 2:
        return False

    return True


def _generate_attempt() -> np.ndarray:
    h = random.randint(17, 22)
    w = random.randint(17, 22)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    source_count = random.randint(2, 3)
    seed_count = random.randint(2, 3)
    colors = random.sample(PALETTE, source_count + seed_count)
    source_colors = colors[:source_count]
    seed_colors = colors[source_count:]
    random.shuffle(seed_colors)

    sources = _place_sources(grid, occupied, source_colors)
    _place_walls(grid, occupied, sources)

    overlap_required = seed_count >= 2 and random.random() < 0.55
    _pick_seeds(grid, occupied, seed_colors, overlap_required)

    if not _validate_generated(grid, overlap_required):
        raise RuntimeError("semantic validation failed")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            return _generate_attempt()
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
