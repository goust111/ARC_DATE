import random
from collections import deque

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 400
GRID_MAX_SIDE = 30
ISLAND_COUNT_RANGE = (3, 5)
OUTER_MARGIN_RANGE = (1, 4)
SIZE_RANGE = (3, 5)
COLORS = tuple(range(1, 10))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = list(map(tuple, np.argwhere(mask)))
    if not coords:
        return []
    seen = {coords[0]}
    q = deque([coords[0]])
    out = []
    while q:
        r, c = q.popleft()
        out.append((int(r), int(c)))
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if (
                0 <= nr < mask.shape[0]
                and 0 <= nc < mask.shape[1]
                and bool(mask[nr, nc])
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                q.append((nr, nc))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_component_cells(mask)) == int(coords.shape[0])


def _make_blob(rng: random.Random) -> np.ndarray:
    h = rng.randint(*SIZE_RANGE)
    w = rng.randint(*SIZE_RANGE)
    mask = np.ones((h, w), dtype=bool)

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    rng.shuffle(corners)
    cut_count = rng.randint(0, 3)
    for r, c in corners[:cut_count]:
        trial = mask.copy()
        trial[r, c] = False
        if _is_connected(trial):
            mask = trial

    return mask


def _find_islands(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    islands = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rc[0] for rc in coords]
            cols = [rc[1] for rc in coords]
            centroid = (
                int(np.floor(np.mean(rows) + 0.5)),
                int(np.floor(np.mean(cols) + 0.5)),
            )
            islands.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "area": len(coords),
                    "centroid": centroid,
                }
            )

    return islands


def _min_manhattan(island_a: dict, island_b: dict) -> int:
    best = 10**9
    for r1, c1 in island_a["coords"]:
        for r2, c2 in island_b["coords"]:
            d = abs(r1 - r2) + abs(c1 - c2)
            if d < best:
                best = d
                if best == 1:
                    return best
    return best


def _analyze_neighbors(grid: np.ndarray) -> tuple[list[dict], list[list[int]], list[list[int]]]:
    islands = _find_islands(grid)
    n = len(islands)
    dists = [[0] * n for _ in range(n)]
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_manhattan(islands[i], islands[j])
            dists[i][j] = dists[j][i] = d
            if d == 3:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return islands, dists, neighbors


def _growth_direction(src: dict, dst: dict) -> tuple[int, int]:
    sr, sc = src["centroid"]
    tr, tc = dst["centroid"]
    dr = tr - sr
    dc = tc - sc
    if abs(dr) > abs(dc):
        return (1, 0) if dr > 0 else (-1, 0)
    return (0, 1) if dc > 0 else (0, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each colored island grows one layer toward its unique largest neighbor.

    Islands are 4-connected monochrome components on a black background.
    Two islands are neighbors when their minimum Manhattan distance is 3,
    which means exactly two black cells separate their boundaries.
    The new layer is colored with the chosen neighbor's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    islands, _dists, neighbors = _analyze_neighbors(g)
    out = g.copy()
    proposals: dict[tuple[int, int], set[int]] = {}

    for idx, island in enumerate(islands):
        if not neighbors[idx]:
            continue

        neighbor_areas = [islands[j]["area"] for j in neighbors[idx]]
        max_area = max(neighbor_areas)
        largest = [j for j in neighbors[idx] if islands[j]["area"] == max_area]
        if len(largest) != 1:
            continue

        target = islands[largest[0]]
        grow_dr, grow_dc = _growth_direction(island, target)
        new_color = int(target["color"])

        for r, c in island["coords"]:
            nr = r + grow_dr
            nc = c + grow_dc
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == 0:
                proposals.setdefault((nr, nc), set()).add(new_color)

    for cell, colors in proposals.items():
        if len(colors) == 1:
            out[cell] = next(iter(colors))

    return out


def _bbox_gap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> int:
    ar, ac, ah, aw = box_a
    br, bc, bh, bw = box_b
    a_r0, a_r1 = ar, ar + ah - 1
    a_c0, a_c1 = ac, ac + aw - 1
    b_r0, b_r1 = br, br + bh - 1
    b_c0, b_c1 = bc, bc + bw - 1

    row_gap = max(0, b_r0 - a_r1, a_r0 - b_r1)
    col_gap = max(0, b_c0 - a_c1, a_c0 - b_c1)
    return row_gap + col_gap


def _place_boxes(rng: random.Random, masks: list[np.ndarray]) -> list[tuple[int, int, int, int]] | None:
    boxes = [(0, 0, masks[0].shape[0], masks[0].shape[1])]
    prev_dir: tuple[int, int] | None = None

    for i in range(1, len(masks)):
        h, w = masks[i].shape
        pr, pc, ph, pw = boxes[-1]

        dirs = list(DIRS4)
        rng.shuffle(dirs)
        if prev_dir is not None:
            opposite = (-prev_dir[0], -prev_dir[1])
            dirs.sort(key=lambda d: d == opposite)

        placed = None
        for dr, dc in dirs:
            for _ in range(24):
                if dr == 0:
                    overlap = 2
                    r_lo = pr - h + overlap
                    r_hi = pr + ph - overlap
                    if r_lo > r_hi:
                        continue
                    nr = rng.randint(r_lo, r_hi)
                    nc = pc + pw + 2 if dc > 0 else pc - w - 2
                else:
                    overlap = 2
                    c_lo = pc - w + overlap
                    c_hi = pc + pw - overlap
                    if c_lo > c_hi:
                        continue
                    nc = rng.randint(c_lo, c_hi)
                    nr = pr + ph + 2 if dr > 0 else pr - h - 2

                candidate = (nr, nc, h, w)
                ok = True
                for j, old in enumerate(boxes):
                    gap = _bbox_gap(candidate, old)
                    if j == len(boxes) - 1:
                        if gap != 3:
                            ok = False
                            break
                    elif gap < 4:
                        ok = False
                        break
                if ok:
                    placed = candidate
                    prev_dir = (dr, dc)
                    break
            if placed is not None:
                break

        if placed is None:
            return None
        boxes.append(placed)

    return boxes


def _graph_connected(neighbors: list[list[int]]) -> bool:
    if not neighbors:
        return False
    seen = {0}
    q = deque([0])
    while q:
        i = q.popleft()
        for j in neighbors[i]:
            if j not in seen:
                seen.add(j)
                q.append(j)
    return len(seen) == len(neighbors)


def _valid_generated_grid(grid: np.ndarray, expected_count: int) -> bool:
    islands, dists, neighbors = _analyze_neighbors(grid)
    if len(islands) != expected_count:
        return False

    areas = [island["area"] for island in islands]
    if len(set(areas)) != len(areas):
        return False

    for i in range(len(islands)):
        for j in range(i + 1, len(islands)):
            if dists[i][j] < 3:
                return False

    edge_count = sum(len(row) for row in neighbors) // 2
    if edge_count != expected_count - 1:
        return False
    if not _graph_connected(neighbors):
        return False
    if max(len(row) for row in neighbors) > 2:
        return False
    if min(len(row) for row in neighbors) < 1:
        return False

    changed = False
    for idx, island in enumerate(islands):
        if not neighbors[idx]:
            return False

        target_idx = max(neighbors[idx], key=lambda j: islands[j]["area"])
        target = islands[target_idx]
        sr, sc = island["centroid"]
        tr, tc = target["centroid"]
        if abs(tr - sr) == abs(tc - sc):
            return False

        dr, dc = _growth_direction(island, target)
        can_grow = False
        for r, c in island["coords"]:
            nr = r + dr
            nc = c + dc
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == 0:
                can_grow = True
                changed = True
                break
        if not can_grow:
            return False

    if not changed:
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(*ISLAND_COUNT_RANGE)

    masks = []
    areas = set()
    for _ in range(count):
        for _ in range(40):
            mask = _make_blob(rng)
            area = int(mask.sum())
            if area not in areas:
                areas.add(area)
                masks.append(mask)
                break
        else:
            return None

    boxes = _place_boxes(rng, masks)
    if boxes is None:
        return None

    min_r = min(r for r, _c, _h, _w in boxes)
    min_c = min(c for _r, c, _h, _w in boxes)
    max_r = max(r + h - 1 for r, _c, h, _w in boxes)
    max_c = max(c + w - 1 for _r, c, _h, w in boxes)

    top = rng.randint(*OUTER_MARGIN_RANGE)
    bottom = rng.randint(*OUTER_MARGIN_RANGE)
    left = rng.randint(*OUTER_MARGIN_RANGE)
    right = rng.randint(*OUTER_MARGIN_RANGE)

    grid_h = (max_r - min_r + 1) + top + bottom
    grid_w = (max_c - min_c + 1) + left + right
    if grid_h > GRID_MAX_SIDE or grid_w > GRID_MAX_SIDE:
        return None

    shift_r = top - min_r
    shift_c = left - min_c
    grid = np.zeros((grid_h, grid_w), dtype=int)
    colors = rng.sample(COLORS, count)

    for color, mask, (r, c, _h, _w) in zip(colors, masks, boxes):
        rr, cc = np.where(mask)
        grid[rr + r + shift_r, cc + c + shift_c] = color

    if not _valid_generated_grid(grid, count):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
