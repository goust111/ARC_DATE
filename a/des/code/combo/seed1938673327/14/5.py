from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 28)
SHORT_CONTACTS = (1, 2)
LONG_CONTACTS = (3, 4, 5)
OPTIONAL_PAIR_PROB = 0.55
SINGLETON_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 500
LOCAL_SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 160


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _extract_objects(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], np.ndarray]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    objects: list[dict[str, object]] = []

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        for coords in _components(grid == color):
            idx = len(objects)
            labels[coords[:, 0], coords[:, 1]] = idx
            anchor = min((int(r), int(c)) for r, c in coords)
            objects.append(
                {
                    "coords": coords,
                    "color": int(color),
                    "area": int(coords.shape[0]),
                    "anchor": anchor,
                }
            )
    return objects, labels


def _adjacency_info(labels: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    h, w = labels.shape
    info: dict[tuple[int, int], dict[str, object]] = {}

    for r in range(h):
        for c in range(w):
            a = int(labels[r, c])
            if a < 0:
                continue
            if c + 1 < w:
                b = int(labels[r, c + 1])
                if b >= 0 and b != a:
                    key = (a, b) if a < b else (b, a)
                    entry = info.setdefault(key, {"count": 0, "pixels": set()})
                    entry["count"] = int(entry["count"]) + 1
                    entry["pixels"].add((r, c))
                    entry["pixels"].add((r, c + 1))
            if r + 1 < h:
                b = int(labels[r + 1, c])
                if b >= 0 and b != a:
                    key = (a, b) if a < b else (b, a)
                    entry = info.setdefault(key, {"count": 0, "pixels": set()})
                    entry["count"] = int(entry["count"]) + 1
                    entry["pixels"].add((r, c))
                    entry["pixels"].add((r + 1, c))
    return info


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge short contacts and mark long contacts with yellow boundary pixels.

    Objects are monochrome 4-connected components. For every pair of adjacent
    objects, the count of touching 4-neighbor pixel pairs decides the action:
    short contacts (<= 2) merge by recoloring the smaller object to the larger
    object's color, while long contacts (>= 3) are highlighted in yellow.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    objects, labels = _extract_objects(g, bg)
    if not objects:
        return g.copy()

    adjacency = _adjacency_info(labels)
    parent = list(range(len(objects)))

    # Merge all short-contact chains by promoting the largest member's color.
    for (a, b), entry in adjacency.items():
        if int(entry["count"]) <= 2:
            _union(parent, a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(objects)):
        groups[_find(parent, idx)].append(idx)

    out = g.copy()
    for members in groups.values():
        if len(members) == 1:
            continue
        leader = min(
            members,
            key=lambda idx: (
                -int(objects[idx]["area"]),
                int(objects[idx]["anchor"][0]),
                int(objects[idx]["anchor"][1]),
                idx,
            ),
        )
        color = int(objects[leader]["color"])
        for idx in members:
            coords = objects[idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = color

    for entry in adjacency.values():
        if int(entry["count"]) >= 3:
            for r, c in entry["pixels"]:
                out[int(r), int(c)] = YELLOW

    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _normalize_pair(
    a_cells: set[tuple[int, int]], b_cells: set[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    min_r = min(min(r for r, _ in a_cells), min(r for r, _ in b_cells))
    min_c = min(min(c for _, c in a_cells), min(c for _, c in b_cells))
    a_norm = sorted((r - min_r, c - min_c) for r, c in a_cells)
    b_norm = sorted((r - min_r, c - min_c) for r, c in b_cells)
    return a_norm, b_norm


def _bbox_shape(cells: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _flip_cells(cells: list[tuple[int, int]], flip_r: bool, flip_c: bool) -> list[tuple[int, int]]:
    h, w = _bbox_shape(cells)
    out = set()
    for r, c in cells:
        nr = h - 1 - r if flip_r else r
        nc = w - 1 - c if flip_c else c
        out.add((nr, nc))
    return _normalize(out)


def _flip_pair(
    a_cells: list[tuple[int, int]],
    b_cells: list[tuple[int, int]],
    flip_r: bool,
    flip_c: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    union = list(a_cells) + list(b_cells)
    h, w = _bbox_shape(union)
    a_out = set()
    b_out = set()
    for r, c in a_cells:
        nr = h - 1 - r if flip_r else r
        nc = w - 1 - c if flip_c else c
        a_out.add((nr, nc))
    for r, c in b_cells:
        nr = h - 1 - r if flip_r else r
        nc = w - 1 - c if flip_c else c
        b_out.add((nr, nc))
    return _normalize_pair(a_out, b_out)


def _shared_boundary_length_local(
    a_cells: list[tuple[int, int]], b_cells: list[tuple[int, int]]
) -> int:
    aset = set(a_cells)
    bset = set(b_cells)
    total = 0
    for r, c in aset:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in bset:
                total += 1
    return total


def _touching_pair(contact_len: int, rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(LOCAL_SHAPE_RETRIES):
        orientation = rng.choice(("horizontal", "vertical"))
        if orientation == "horizontal":
            h1 = contact_len + rng.randint(0, 3)
            h2 = contact_len + rng.randint(0, 3)
            w1 = rng.randint(2, 5)
            w2 = rng.randint(2, 5)
            offset = contact_len - h2 if rng.random() < 0.5 else h1 - contact_len
            a = {(r, c) for r in range(h1) for c in range(w1)}
            b = {(offset + r, w1 + c) for r in range(h2) for c in range(w2)}
        else:
            w1 = contact_len + rng.randint(0, 3)
            w2 = contact_len + rng.randint(0, 3)
            h1 = rng.randint(2, 5)
            h2 = rng.randint(2, 5)
            offset = contact_len - w2 if rng.random() < 0.5 else w1 - contact_len
            a = {(r, c) for r in range(h1) for c in range(w1)}
            b = {(h1 + r, offset + c) for r in range(h2) for c in range(w2)}

        a_cells, b_cells = _normalize_pair(a, b)
        if rng.random() < 0.5:
            a_cells, b_cells = _flip_pair(a_cells, b_cells, flip_r=True, flip_c=False)
        if rng.random() < 0.5:
            a_cells, b_cells = _flip_pair(a_cells, b_cells, flip_r=False, flip_c=True)
        if _shared_boundary_length_local(a_cells, b_cells) != contact_len:
            continue
        if contact_len <= 2 and len(a_cells) == len(b_cells):
            continue
        return a_cells, b_cells
    return None


def _random_polyomino(rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(LOCAL_SHAPE_RETRIES):
        target = rng.randint(3, 7)
        cells = {(0, 0)}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)
            cells.add(rng.choice(list(frontier)))
        norm = _normalize(cells)
        h, w = _bbox_shape(norm)
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue
        if rng.random() < 0.5:
            norm = _flip_cells(norm, flip_r=True, flip_c=False)
        if rng.random() < 0.5:
            norm = _flip_cells(norm, flip_r=False, flip_c=True)
        return norm
    return None


def _cluster_bbox(cluster: list[list[tuple[int, int]]]) -> tuple[int, int]:
    union = {cell for shape in cluster for cell in shape}
    return _bbox_shape(_normalize(union))


def _can_place(
    blocked: np.ndarray, cells: list[tuple[int, int]], top: int, left: int
) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        rr = top + r
        cc = left + c
        if rr < 0 or rr >= h or cc < 0 or cc >= w or blocked[rr, cc]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], top: int, left: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        rr = top + r
        cc = left + c
        blocked[rr, cc] = True
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], top: int, left: int, color: int) -> None:
    for r, c in cells:
        grid[top + r, left + c] = color


def _place_cluster(
    grid: np.ndarray,
    blocked: np.ndarray,
    cluster: list[list[tuple[int, int]]],
    colors: list[int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    box_h, box_w = _cluster_bbox(cluster)
    if box_h > h or box_w > w:
        return False

    union = _normalize({cell for shape in cluster for cell in shape})
    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(0, h - box_h)
        left = rng.randint(0, w - box_w)
        if not _can_place(blocked, union, top, left):
            continue
        for shape, color in zip(cluster, colors):
            _paint(grid, shape, top, left, color)
        _mark_blocked(blocked, union, top, left)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    pair_contacts = [rng.choice(SHORT_CONTACTS), rng.choice(LONG_CONTACTS)]
    if rng.random() < OPTIONAL_PAIR_PROB:
        extra_pool = SHORT_CONTACTS + LONG_CONTACTS
        pair_contacts.append(rng.choice(extra_pool))

    singleton_count = rng.randint(*SINGLETON_RANGE)
    total_objects = 2 * len(pair_contacts) + singleton_count
    if total_objects > len(PALETTE):
        singleton_count = len(PALETTE) - 2 * len(pair_contacts)
        total_objects = 2 * len(pair_contacts) + singleton_count
    if total_objects <= 0:
        return None

    clusters: list[list[list[tuple[int, int]]]] = []
    for contact_len in pair_contacts:
        pair = _touching_pair(int(contact_len), rng)
        if pair is None:
            return None
        clusters.append([pair[0], pair[1]])

    for _ in range(singleton_count):
        shape = _random_polyomino(rng)
        if shape is None:
            return None
        clusters.append([shape])

    colors = rng.sample(list(PALETTE), total_objects)
    colored_clusters: list[tuple[list[list[tuple[int, int]]], list[int]]] = []
    cursor = 0
    for cluster in clusters:
        size = len(cluster)
        colored_clusters.append((cluster, colors[cursor : cursor + size]))
        cursor += size

    rng.shuffle(colored_clusters)
    colored_clusters.sort(key=lambda item: (_cluster_bbox(item[0])[0] * _cluster_bbox(item[0])[1]), reverse=True)

    for cluster, cluster_colors in colored_clusters:
        if not _place_cluster(grid, blocked, cluster, cluster_colors, rng):
            return None

    if _mode_color(grid) != BG:
        return None

    objects, labels = _extract_objects(grid, BG)
    if len(objects) != total_objects:
        return None
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return None

    adjacency = _adjacency_info(labels)
    counts = [int(entry["count"]) for entry in adjacency.values()]
    if not counts or not any(count <= 2 for count in counts) or not any(count >= 3 for count in counts):
        return None

    degrees: dict[int, int] = defaultdict(int)
    for a, b in adjacency:
        degrees[int(a)] += 1
        degrees[int(b)] += 1
    if any(deg > 1 for deg in degrees.values()):
        return None

    for (a, b), entry in adjacency.items():
        if int(entry["count"]) <= 2 and int(objects[a]["area"]) == int(objects[b]["area"]):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.any(out == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
