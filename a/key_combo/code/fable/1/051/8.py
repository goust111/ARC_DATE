from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


OFFSETS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(10))

LOCAL_SIDE_RANGE = (10, 16)
GRID_SIDE_RANGE = (18, 28)
GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    for dr, dc in OFFSETS4:
        nr, nc = int(r) + dr, int(c) + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_border(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def _bfs_dist(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(mask.shape, -1, dtype=int)
    sr, sc = map(int, start)
    if not bool(mask[sr, sc]):
        return dist
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(mask.shape, r, c):
            if not bool(mask[nr, nc]) or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _count_floor_neighbors(mask: np.ndarray, r: int, c: int) -> int:
    return sum(bool(mask[nr, nc]) for nr, nc in _neighbors4(mask.shape, r, c))


def _has_hole(mask: np.ndarray) -> bool:
    b = _bbox(mask)
    if b is None:
        return False
    r0, c0, r1, c1 = b
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~sub
    border_empty = [
        (r, c)
        for r in range(sub.shape[0])
        for c in range(sub.shape[1])
        if empty[r, c]
        and (r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1)
    ]
    seen = np.zeros_like(empty, dtype=bool)
    q: deque[tuple[int, int]] = deque(border_empty)
    for r, c in border_empty:
        seen[r, c] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(sub.shape, r, c):
            if empty[nr, nc] and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool((empty & ~seen).any())


def _boundary_from_floor(floor_mask: np.ndarray) -> np.ndarray:
    wall = np.zeros_like(floor_mask, dtype=bool)
    for r, c in np.argwhere(floor_mask):
        for nr, nc in _neighbors4(floor_mask.shape, int(r), int(c)):
            if not bool(floor_mask[nr, nc]):
                wall[nr, nc] = True
    return wall


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _pick_weighted(items: list[tuple[int, tuple[int, int], tuple[int, int]]]):
    weights = [item[0] for item in items]
    return random.choices(items, weights=weights, k=1)[0]


def _generate_local_floor() -> np.ndarray | None:
    h = random.randint(*LOCAL_SIDE_RANGE)
    w = random.randint(*LOCAL_SIDE_RANGE)
    mask = np.zeros((h, w), dtype=bool)

    start = (random.randint(1, h - 2), random.randint(1, w - 2))
    mask[start] = True
    cells = [start]
    recent = start
    target_tree = random.randint(max(14, (h + w) // 2), min(h * w // 3, 28))

    while int(mask.sum()) < target_tree:
        choices: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        for anchor in cells:
            ar, ac = anchor
            for nr, nc in _neighbors4(mask.shape, ar, ac):
                if nr in (0, h - 1) or nc in (0, w - 1) or bool(mask[nr, nc]):
                    continue
                if _count_floor_neighbors(mask, nr, nc) != 1:
                    continue
                weight = 1 + (5 if anchor == recent else 0)
                choices.append((weight, anchor, (nr, nc)))
        if not choices:
            break
        _weight, _anchor, chosen = _pick_weighted(choices)
        mask[chosen] = True
        cells.append(chosen)
        recent = chosen

    if int(mask.sum()) < 14:
        return None

    target_area = min(h * w - 4, int(mask.sum()) + random.randint(6, 16))
    guard = 0
    while int(mask.sum()) < target_area and guard < target_area * 12:
        guard += 1
        candidates = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if bool(mask[r, c]):
                    continue
                deg = _count_floor_neighbors(mask, r, c)
                if deg == 0 or deg > 2:
                    continue
                weight = 4 if deg == 1 else 2
                candidates.append((weight, (r, c)))
        if not candidates:
            break
        weights = [weight for weight, _cell in candidates]
        r, c = random.choices([cell for _weight, cell in candidates], weights=weights, k=1)[0]
        mask[r, c] = True
        if _has_hole(mask):
            mask[r, c] = False

    b = _bbox(mask)
    if b is None:
        return None
    r0, c0, r1, c1 = b
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    area = int(mask.sum())
    if area < 18 or bbox_area - area < 5:
        return None
    if _has_hole(mask):
        return None
    return mask


def _embed_floor(local_floor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lh, lw = map(int, local_floor.shape)
    h = random.randint(max(GRID_SIDE_RANGE[0], lh + 6), min(GRID_SIDE_RANGE[1], lh + 12))
    w = random.randint(max(GRID_SIDE_RANGE[0], lw + 6), min(GRID_SIDE_RANGE[1], lw + 12))
    grid_floor = np.zeros((h, w), dtype=bool)
    r0 = random.randint(2, h - lh - 3)
    c0 = random.randint(2, w - lw - 3)
    grid_floor[r0 : r0 + lh, c0 : c0 + lw] = local_floor
    return grid_floor, _boundary_from_floor(grid_floor)


def _candidate_markers(floor_mask: np.ndarray, wall_mask: np.ndarray) -> list[tuple[int, int]]:
    outside = ~(floor_mask | wall_mask)
    markers = []
    for r, c in np.argwhere(wall_mask):
        rr, cc = int(r), int(c)
        touches_floor = False
        touches_outside = False
        for nr, nc in _neighbors4(wall_mask.shape, rr, cc):
            touches_floor |= bool(floor_mask[nr, nc])
            touches_outside |= bool(outside[nr, nc])
        if touches_floor and touches_outside:
            markers.append((rr, cc))
    return markers


def _search_seed_and_markers(
    floor_mask: np.ndarray, wall_mask: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    floor_cells = [tuple(map(int, rc)) for rc in np.argwhere(floor_mask)]
    marker_cells = _candidate_markers(floor_mask, wall_mask)
    if len(marker_cells) < 2:
        return None

    best_score = None
    best_triplet = None

    for seed in floor_cells:
        dist = _bfs_dist(floor_mask, seed)
        reachable = dist >= 0
        if not np.array_equal(reachable, floor_mask):
            continue
        if int(dist.max()) < 5:
            continue
        if not ((dist == 1).any() and (dist >= 2).any()):
            continue
        if not any(
            dist[r, c] > _manhattan(seed, (r, c))
            for r, c in floor_cells
            if (r, c) != seed
        ):
            continue

        infos = []
        for marker in marker_cells:
            neighbor_dists = [
                int(dist[nr, nc])
                for nr, nc in _neighbors4(floor_mask.shape, marker[0], marker[1])
                if bool(floor_mask[nr, nc])
            ]
            if not neighbor_dists:
                continue
            gdist = 1 + min(neighbor_dists)
            mdist = _manhattan(seed, marker)
            edist2 = (seed[0] - marker[0]) ** 2 + (seed[1] - marker[1]) ** 2
            infos.append(
                {
                    "pos": marker,
                    "gdist": gdist,
                    "mdist": mdist,
                    "edist2": edist2,
                }
            )

        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                a = infos[i]
                b = infos[j]
                if a["gdist"] == b["gdist"]:
                    continue
                near = a if a["gdist"] < b["gdist"] else b
                far = b if near is a else a
                if near["mdist"] <= far["mdist"]:
                    continue
                if near["edist2"] <= far["edist2"]:
                    continue
                if _manhattan(near["pos"], far["pos"]) < 4:
                    continue
                score = (
                    5 * (far["gdist"] - near["gdist"])
                    + 2 * (near["mdist"] - far["mdist"])
                    + (near["edist2"] - far["edist2"])
                )
                if best_score is None or score > best_score or (
                    score == best_score and random.random() < 0.5
                ):
                    best_score = score
                    best_triplet = (seed, near["pos"], far["pos"])

    return best_triplet


def _build_input_grid(
    floor_mask: np.ndarray,
    wall_mask: np.ndarray,
    seed: tuple[int, int],
    near_marker: tuple[int, int],
    far_marker: tuple[int, int],
) -> np.ndarray:
    bg, wall_color, seed_color, color_a, color_b = random.sample(PALETTE, 5)
    grid = np.full(floor_mask.shape, bg, dtype=int)
    grid[wall_mask] = wall_color
    grid[seed] = seed_color

    marker_positions = [near_marker, far_marker]
    marker_colors = [color_a, color_b]
    random.shuffle(marker_colors)
    grid[marker_positions[0]] = marker_colors[0]
    grid[marker_positions[1]] = marker_colors[1]
    return grid


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = get_bg(g)
    colors = sorted(set(map(int, g.flatten())) - {bg})
    assert len(colors) == 4, "expected background, wall, seed, and two marker colors"

    counts = Counter(map(int, g.flatten()))
    wall_count, wall_color = max((int(counts[c]), int(c)) for c in colors)
    assert sum(int(counts[c]) == wall_count for c in colors) == 1, "wall color must be uniquely largest"

    singletons = [int(c) for c in colors if int(c) != wall_color]
    assert len(singletons) == 3, "expected three singleton colors"
    for color in singletons:
        assert int(np.sum(g == color)) == 1, "seed and markers must each occupy one cell"

    seed_candidates = []
    for seed_color in singletons:
        seed_pos = tuple(map(int, np.argwhere(g == seed_color)[0]))
        walk_mask = (g == bg) | (g == seed_color)
        dist = _bfs_dist(walk_mask, seed_pos)
        cavity_with_seed = dist >= 0
        if _touches_border(cavity_with_seed):
            continue
        marker_colors = [color for color in singletons if color != seed_color]
        marker_positions = [
            tuple(map(int, np.argwhere(g == marker_color)[0])) for marker_color in marker_colors
        ]
        if not all(any(cavity_with_seed[nr, nc] for nr, nc in _neighbors4(g.shape, *pos)) for pos in marker_positions):
            continue
        seed_candidates.append((seed_color, seed_pos, marker_colors, marker_positions, dist))

    assert len(seed_candidates) == 1, "seed should be uniquely identifiable"
    seed_color, seed_pos, marker_colors, marker_positions, dist = seed_candidates[0]
    cavity_mask = dist >= 0
    floor_bg_mask = cavity_mask & (g == bg)
    wall_mask = g == wall_color
    assert not bool(wall_mask[seed_pos]), "seed cannot be on the wall"
    assert floor_bg_mask.any(), "cavity must contain background cells besides the seed"

    markers = []
    for marker_color, marker_pos in zip(marker_colors, marker_positions):
        neighbor_dists = [
            int(dist[nr, nc])
            for nr, nc in _neighbors4(g.shape, marker_pos[0], marker_pos[1])
            if bool(cavity_mask[nr, nc])
        ]
        assert neighbor_dists, "marker must touch the cavity"
        markers.append(
            {
                "color": int(marker_color),
                "pos": tuple(marker_pos),
                "gdist": 1 + min(neighbor_dists),
                "mdist": _manhattan(seed_pos, marker_pos),
                "edist2": (seed_pos[0] - marker_pos[0]) ** 2 + (seed_pos[1] - marker_pos[1]) ** 2,
            }
        )

    markers.sort(key=lambda item: item["gdist"])
    assert markers[0]["gdist"] != markers[1]["gdist"], "marker geodesic distances must differ"

    return {
        "bg": int(bg),
        "wall_color": int(wall_color),
        "wall_mask": wall_mask,
        "seed_color": int(seed_color),
        "seed_pos": tuple(seed_pos),
        "cavity_mask": cavity_mask,
        "floor_bg_mask": floor_bg_mask,
        "dist": dist,
        "near_marker": markers[0],
        "far_marker": markers[1],
    }


def _paint_with_order(grid: np.ndarray, odd_color: int, even_color: int) -> np.ndarray:
    info = _analyze_grid(grid)
    out = np.asarray(grid, dtype=int).copy()
    d = info["dist"]
    fill = info["floor_bg_mask"]
    out[fill & ((d % 2) == 1)] = int(odd_color)
    out[fill & ((d % 2) == 0)] = int(even_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the enclosed cavity from the seed with alternating geodesic rings colored by wall markers.

    Input structure:
    - One wall color encloses a single irregular cavity whose interior cells are otherwise background.
    - The cavity contains one seed pixel of a unique color.
    - Exactly two wall cells are recolored as different marker colors, and both markers remain on the wall.

    Rule:
    - Identify the cavity reachable from the seed using only 4-connected moves through cavity cells.
    - For each marker, compare its cavity-side shortest-path distance from the seed by looking at the nearest adjacent cavity cell.
    - Keep the seed unchanged; recolor every other cavity cell with the nearer marker's color at odd distance and the farther marker's color at even distance, while leaving the wall and markers unchanged.

    Key insight:
    - The cavity fill is an odd/even wavefront from the seed, but deciding which marker color applies first requires comparing the markers by geodesic distance through the cavity rather than by straight-line closeness.
    """
    info = _analyze_grid(grid)
    return _paint_with_order(
        grid,
        odd_color=info["near_marker"]["color"],
        even_color=info["far_marker"]["color"],
    )


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False
    d = info["dist"]
    fill = info["floor_bg_mask"]

    if not bool((fill & ((d % 2) == 1)).any()):
        return False
    if not bool((fill & ((d % 2) == 0)).any()):
        return False

    floor_cells = np.argwhere(info["cavity_mask"])
    if not any(
        int(d[r, c]) > _manhattan(info["seed_pos"], (int(r), int(c)))
        for r, c in floor_cells
        if int(d[r, c]) >= 0 and (int(r), int(c)) != info["seed_pos"]
    ):
        return False

    near = info["near_marker"]
    far = info["far_marker"]
    if not (near["gdist"] < far["gdist"]):
        return False
    if not (near["mdist"] > far["mdist"]):
        return False
    if not (near["edist2"] > far["edist2"]):
        return False

    correct = main(grid)
    manhattan_order = sorted([near, far], key=lambda item: item["mdist"])
    if manhattan_order[0]["color"] == near["color"] and manhattan_order[1]["color"] == far["color"]:
        return False
    wrong = _paint_with_order(
        grid,
        odd_color=manhattan_order[0]["color"],
        even_color=manhattan_order[1]["color"],
    )
    if np.array_equal(correct, wrong):
        return False

    if np.array_equal(correct, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        local_floor = _generate_local_floor()
        if local_floor is None:
            continue
        floor_mask, wall_mask = _embed_floor(local_floor)
        triplet = _search_seed_and_markers(floor_mask, wall_mask)
        if triplet is None:
            continue
        seed, near_marker, far_marker = triplet
        grid = _build_input_grid(floor_mask, wall_mask, seed, near_marker, far_marker)
        if _validate_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid geodesic-rings instance")
