from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_objects(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    owner = np.full((h, w), -1, dtype=int)
    objects: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            idx = len(objects)
            owner[arr[:, 0], arr[:, 1]] = idx
            objects.append(arr)

    return objects, owner


def _neighbor_counts(objects: list[np.ndarray], owner: np.ndarray) -> list[int]:
    h, w = owner.shape
    counts: list[int] = []
    for idx, coords in enumerate(objects):
        neighbors: set[int] = set()
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    other = int(owner[rr, cc])
                    if other >= 0 and other != idx:
                        neighbors.add(other)
        counts.append(len(neighbors))
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each foreground object using the bottom lookup row indexed by neighbor count."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 1, "expected a 2D grid with table rows"
    assert np.all(g[-2] == g[-2, 0]), "expected a full-width separator row above the palette"

    scene = g[:-2]
    palette = g[-1]
    objects, owner = _find_objects(scene)
    counts = _neighbor_counts(objects, owner)

    out = np.full(scene.shape, BG, dtype=int)
    for coords, degree in zip(objects, counts, strict=True):
        assert degree < palette.shape[0], "palette row must cover every neighbor count"
        out[coords[:, 0], coords[:, 1]] = int(palette[degree])
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
COARSE_H_RANGE = (4, 7)
COARSE_W_RANGE = (5, 8)
SCALE_OPTIONS = (2, 3)
CHAIN_LEN_RANGE = (3, 5)
ISOLATE_COUNT_RANGE = (0, 2)
HUB_LEAF_RANGE = (3, 4)
RANDOM_OBJECT_COUNT_RANGE = (4, 7)
RANDOM_OBJECT_SIZE_RANGE = (1, 3)
SAFE_EXTRA_CELLS = 2
MODES = ("chain", "hub", "random")


def _empty_candidates(labels: np.ndarray, *, require_isolated: bool) -> list[tuple[int, int]]:
    h, w = labels.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if int(labels[r, c]) != -1:
                continue
            if require_isolated:
                ok = True
                for dr, dc in DIRS4:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and int(labels[rr, cc]) != -1:
                        ok = False
                        break
                if not ok:
                    continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _place_isolates(labels: np.ndarray, next_label: int, count: int) -> int:
    placed = 0
    for r, c in _empty_candidates(labels, require_isolated=True):
        if placed >= count:
            break
        labels[r, c] = next_label + placed
        placed += 1
    return placed


def _frontier_cells(labels: np.ndarray, label: int, *, preserve_adjacency: bool) -> list[tuple[int, int]]:
    h, w = labels.shape
    frontier: set[tuple[int, int]] = set()
    for r_raw, c_raw in np.argwhere(labels == label):
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w) or int(labels[rr, cc]) != -1:
                continue
            if preserve_adjacency:
                ok = True
                for dr2, dc2 in DIRS4:
                    nr = rr + dr2
                    nc = cc + dc2
                    if 0 <= nr < h and 0 <= nc < w:
                        other = int(labels[nr, nc])
                        if other not in {-1, label}:
                            ok = False
                            break
                if not ok:
                    continue
            frontier.add((rr, cc))
    cells = list(frontier)
    random.shuffle(cells)
    return cells


def _expand_labels(labels: np.ndarray, *, preserve_adjacency: bool) -> None:
    object_ids = [int(v) for v in np.unique(labels) if int(v) >= 0]
    random.shuffle(object_ids)
    for label in object_ids:
        for _ in range(random.randint(0, SAFE_EXTRA_CELLS)):
            frontier = _frontier_cells(labels, label, preserve_adjacency=preserve_adjacency)
            if not frontier:
                break
            r, c = frontier[0]
            labels[r, c] = label


def _chain_labels(h: int, w: int) -> np.ndarray | None:
    labels = np.full((h, w), -1, dtype=int)
    horizontal = random.choice([False, True])
    max_len = min(CHAIN_LEN_RANGE[1], w if horizontal else h)
    min_len = min(CHAIN_LEN_RANGE[0], max_len)
    if max_len < CHAIN_LEN_RANGE[0]:
        return None
    length = random.randint(min_len, max_len)

    if horizontal:
        r = random.randrange(h)
        c0 = random.randrange(w - length + 1)
        for i in range(length):
            labels[r, c0 + i] = i
    else:
        c = random.randrange(w)
        r0 = random.randrange(h - length + 1)
        for i in range(length):
            labels[r0 + i, c] = i

    next_label = length
    isolate_target = random.randint(*ISOLATE_COUNT_RANGE)
    next_label += _place_isolates(labels, next_label, isolate_target)
    _expand_labels(labels, preserve_adjacency=True)
    return labels


def _hub_labels(h: int, w: int) -> np.ndarray | None:
    if h < 3 or w < 3:
        return None

    labels = np.full((h, w), -1, dtype=int)
    center_r = random.randrange(1, h - 1)
    center_c = random.randrange(1, w - 1)
    labels[center_r, center_c] = 0

    dirs = list(DIRS4)
    random.shuffle(dirs)
    leaf_count = random.randint(HUB_LEAF_RANGE[0], min(HUB_LEAF_RANGE[1], len(dirs)))
    for idx, (dr, dc) in enumerate(dirs[:leaf_count], start=1):
        labels[center_r + dr, center_c + dc] = idx

    next_label = leaf_count + 1
    isolate_target = random.randint(*ISOLATE_COUNT_RANGE)
    next_label += _place_isolates(labels, next_label, isolate_target)
    _expand_labels(labels, preserve_adjacency=True)
    return labels


def _random_labels(h: int, w: int) -> np.ndarray | None:
    labels = np.full((h, w), -1, dtype=int)
    max_objects = min(RANDOM_OBJECT_COUNT_RANGE[1], h * w)
    if max_objects < RANDOM_OBJECT_COUNT_RANGE[0]:
        return None
    n_objects = random.randint(RANDOM_OBJECT_COUNT_RANGE[0], max_objects)

    seeds = random.sample([(r, c) for r in range(h) for c in range(w)], n_objects)
    sizes = [random.randint(*RANDOM_OBJECT_SIZE_RANGE) for _ in range(n_objects)]
    filled = [1] * n_objects

    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx

    active = [idx for idx, target in enumerate(sizes) if target > 1]
    while active:
        idx = random.choice(active)
        frontier = _frontier_cells(labels, idx, preserve_adjacency=False)
        if frontier:
            r, c = random.choice(frontier)
            labels[r, c] = idx
            filled[idx] += 1
        if filled[idx] >= sizes[idx] or not frontier:
            active = [j for j in active if filled[j] < sizes[j] and _frontier_cells(labels, j, preserve_adjacency=False)]

    return labels


def _label_neighbor_counts(labels: np.ndarray) -> dict[int, int]:
    counts: dict[int, set[int]] = {}
    h, w = labels.shape
    for r in range(h):
        for c in range(w):
            label = int(labels[r, c])
            if label < 0:
                continue
            counts.setdefault(label, set())
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    other = int(labels[rr, cc])
                    if other >= 0 and other != label:
                        counts[label].add(other)
    return {label: len(neighbors) for label, neighbors in counts.items()}


def _paint_scene(labels: np.ndarray, scale: int, object_colors: dict[int, int]) -> np.ndarray:
    h, w = labels.shape
    scene = np.full((h * scale, w * scale), BG, dtype=int)
    for r in range(h):
        for c in range(w):
            label = int(labels[r, c])
            if label < 0:
                continue
            color = object_colors[label]
            scene[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = color
    return scene


def _build_palette(width: int, used_counts: list[int]) -> np.ndarray:
    palette = np.array([random.choice(COLOR_POOL) for _ in range(width)], dtype=int)
    used_colors = random.sample(COLOR_POOL, len(used_counts))
    for count, color in zip(used_counts, used_colors, strict=True):
        palette[count] = color
    return palette


def _expected_scene(labels: np.ndarray, scale: int, palette: np.ndarray, degrees: dict[int, int]) -> np.ndarray:
    h, w = labels.shape
    out = np.full((h * scale, w * scale), BG, dtype=int)
    for r in range(h):
        for c in range(w):
            label = int(labels[r, c])
            if label < 0:
                continue
            out[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = int(palette[degrees[label]])
    return out


def _gen_attempt() -> np.ndarray | None:
    coarse_h = random.randint(*COARSE_H_RANGE)
    coarse_w = random.randint(*COARSE_W_RANGE)
    mode = random.choice(MODES)

    if mode == "chain":
        labels = _chain_labels(coarse_h, coarse_w)
    elif mode == "hub":
        labels = _hub_labels(coarse_h, coarse_w)
    else:
        labels = _random_labels(coarse_h, coarse_w)
    if labels is None:
        return None

    degrees = _label_neighbor_counts(labels)
    if len(degrees) < 3:
        return None

    used_counts = sorted(set(degrees.values()))
    if len(used_counts) < 2 or max(used_counts) == 0:
        return None

    scale = random.choice(SCALE_OPTIONS)
    width = coarse_w * scale
    if max(used_counts) >= width:
        return None

    palette = _build_palette(width, used_counts)
    object_ids = sorted(degrees)
    object_colors = {label: color for label, color in zip(object_ids, random.sample(COLOR_POOL, len(object_ids)), strict=True)}
    if all(object_colors[label] == int(palette[degrees[label]]) for label in object_ids):
        rotated = random.sample(COLOR_POOL, len(object_ids))
        rotated = rotated[1:] + rotated[:1]
        object_colors = {label: color for label, color in zip(object_ids, rotated, strict=True)}

    scene = _paint_scene(labels, scale, object_colors)
    expected = _expected_scene(labels, scale, palette, degrees)
    if np.array_equal(scene, expected):
        return None

    separator = np.full((1, width), SEPARATOR_COLOR, dtype=int)
    inp = np.vstack([scene, separator, palette[None, :]])

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if np.array_equal(solved, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
