import random
from collections import Counter, defaultdict, deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- helpers ---------------


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _label_regions(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    labels = np.zeros((h, w), dtype=int)
    label_to_color: dict[int, int] = {}
    label_to_cells: dict[int, list[tuple[int, int]]] = {}
    next_label = 1
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or labels[r, c] != 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = next_label
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and labels[nr, nc] == 0
                        and int(grid[nr, nc]) == color
                    ):
                        labels[nr, nc] = next_label
                        q.append((nr, nc))
            label_to_color[next_label] = color
            label_to_cells[next_label] = cells
            next_label += 1
    return labels, label_to_color, label_to_cells


def _adjacency_counts(labels: np.ndarray) -> dict[int, Counter]:
    h, w = labels.shape
    adj: dict[int, Counter] = defaultdict(Counter)
    for r in range(h):
        for c in range(w):
            a = int(labels[r, c])
            if a == 0:
                continue
            if r + 1 < h:
                b = int(labels[r + 1, c])
                if b != 0 and b != a:
                    adj[a][b] += 1
                    adj[b][a] += 1
            if c + 1 < w:
                b = int(labels[r, c + 1])
                if b != 0 and b != a:
                    adj[a][b] += 1
                    adj[b][a] += 1
    return adj


def _winner_neighbor(adj: dict[int, Counter], region_id: int) -> int | None:
    if region_id not in adj or not adj[region_id]:
        return None
    items = list(adj[region_id].items())
    items.sort(key=lambda x: (-x[1], x[0]))
    return int(items[0][0])


def _hole_fill_plan(grid: np.ndarray) -> list[tuple[list[tuple[int, int]], int]]:
    labels, label_to_color, _ = _label_regions(grid)
    adj = _adjacency_counts(labels)
    h, w = grid.shape
    plans: list[tuple[list[tuple[int, int]], int]] = []
    for comp in _components_from_mask(grid == 0):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        neighbor_labels: set[int] = set()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] != 0:
                    neighbor_labels.add(int(labels[nr, nc]))
        if len(neighbor_labels) != 1:
            continue
        region_id = next(iter(neighbor_labels))
        winner = _winner_neighbor(adj, region_id)
        if winner is None:
            continue
        plans.append((comp, int(label_to_color[winner])))
    return plans


def _connected_after_cut(region_mask: np.ndarray, hole_cells: list[tuple[int, int]]) -> bool:
    trimmed = region_mask.copy()
    for r, c in hole_cells:
        trimmed[r, c] = False
    comps = _components_from_mask(trimmed)
    return len(comps) == 1


def _pick_seeds(h: int, w: int, k: int, rng: random.Random) -> list[tuple[int, int]]:
    seeds: list[tuple[int, int]] = []
    min_dist = max(3, min(h, w) // 4)
    for _ in range(200):
        seeds.clear()
        points = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(points)
        for pt in points:
            if all(abs(pt[0] - sr) + abs(pt[1] - sc) >= min_dist for sr, sc in seeds):
                seeds.append(pt)
                if len(seeds) == k:
                    return seeds.copy()
        min_dist = max(2, min_dist - 1)
    raise RuntimeError("failed to place seeds")


def _grow_partition(h: int, w: int, k: int, rng: random.Random) -> np.ndarray:
    seeds = _pick_seeds(h, w, k, rng)
    owner = np.full((h, w), -1, dtype=int)
    frontier: list[list[tuple[int, int]]] = [[] for _ in range(k)]
    sizes = [0] * k
    for rid, (r, c) in enumerate(seeds):
        owner[r, c] = rid
        frontier[rid].append((r, c))
        sizes[rid] = 1

    total = h * w
    target = total / k
    assigned = k
    while assigned < total:
        active = [rid for rid in range(k) if frontier[rid]]
        if not active:
            raise RuntimeError("partition growth stalled")
        weights = [max(1.0, target / max(1, sizes[rid])) for rid in active]
        rid = rng.choices(active, weights=weights, k=1)[0]
        idx = rng.randrange(len(frontier[rid]))
        r, c = frontier[rid][idx]
        neighbors = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and owner[nr, nc] == -1:
                neighbors.append((nr, nc))
        if not neighbors:
            frontier[rid][idx] = frontier[rid][-1]
            frontier[rid].pop()
            continue
        nr, nc = rng.choice(neighbors)
        owner[nr, nc] = rid
        frontier[rid].append((nr, nc))
        sizes[rid] += 1
        assigned += 1
        if len(neighbors) == 1:
            frontier[rid][idx] = frontier[rid][-1]
            frontier[rid].pop()
    return owner


HOLE_SHAPES = [
    [(0, 0)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
]


def _try_cut_hole(grid: np.ndarray, region_color: int, rng: random.Random) -> bool:
    region_mask = grid == region_color
    cells = np.argwhere(region_mask)
    shapes = HOLE_SHAPES[:]
    rng.shuffle(shapes)
    cell_order = [(int(r), int(c)) for r, c in cells]
    rng.shuffle(cell_order)
    h, w = grid.shape
    for shape in shapes:
        for ar, ac in cell_order:
            coords = [(ar + dr, ac + dc) for dr, dc in shape]
            if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in coords):
                continue
            if any(grid[r, c] != region_color for r, c in coords):
                continue
            hole_set = set(coords)
            ok = True
            for r, c in coords:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in hole_set:
                        continue
                    if grid[nr, nc] != region_color:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or not _connected_after_cut(region_mask, coords):
                continue
            for r, c in coords:
                grid[r, c] = 0
            return True
    return False


def _build_candidate_grid(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 22)
    w = rng.randint(12, 22)
    k = rng.randint(4, min(7, 9))
    owner = _grow_partition(h, w, k, rng)
    sizes = Counter(owner.flatten().tolist())
    if min(sizes.values()) < max(14, (h * w) // (3 * k)):
        return None

    colors = rng.sample(list(range(1, 10)), k)
    grid = np.zeros((h, w), dtype=int)
    for rid, color in enumerate(colors):
        grid[owner == rid] = color

    labels, label_to_color, cells_by_label = _label_regions(grid)
    adj = _adjacency_counts(labels)
    candidates = []
    for label, cells in cells_by_label.items():
        if len(cells) < 18 or len(adj[label]) < 2:
            continue
        counts = list(adj[label].values())
        if counts.count(max(counts)) != 1:
            continue
        candidates.append(label)
    if not candidates:
        return None

    rng.shuffle(candidates)
    target_holes = min(len(candidates), rng.randint(1, 3))
    placed = 0
    for label in candidates:
        if _try_cut_hole(grid, label_to_color[label], rng):
            placed += 1
        if placed == target_holes:
            break
    if placed == 0:
        return None

    plans = _hole_fill_plan(grid)
    if len(plans) != placed:
        return None
    for comp, fill_color in plans:
        around = set()
        for r, c in comp:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0:
                    around.add(int(grid[nr, nc]))
        if len(around) != 1 or fill_color in around:
            return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


# --------------- main solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black hole with the color of the neighboring region that
    has the longest shared boundary with the surrounding region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for comp, fill_color in _hole_fill_plan(g):
        for r, c in comp:
            out[r, c] = fill_color
    return out


# --------------- generator ---------------


MAX_ATTEMPTS = 300


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
