import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

BG = 0
MAX_GENERATE_ATTEMPTS = 400


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> list[tuple[int, int]]:
    dirs = DIR4 if connectivity == 4 else DIR8
    out = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _label_mask(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label += 1
            q = deque([(r, c)])
            labels[r, c] = label
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label
                        q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return labels, components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep every object that belongs to the marker's 8-connected foreground region.
    Replace every other object by the hollow frame of its tight bounding box.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    counts = Counter(map(int, g.ravel()))

    marker_colors = [color for color, count in counts.items() if color != bg and count == 1]
    assert len(marker_colors) == 1, "expected a unique singleton marker color"
    marker_color = int(marker_colors[0])
    marker_r, marker_c = map(int, np.argwhere(g == marker_color)[0])

    fg_labels8, _ = _label_mask(g != bg, connectivity=8)
    marker_component = int(fg_labels8[marker_r, marker_c])
    assert marker_component != 0

    out = g.copy()
    for color in sorted(set(map(int, g.ravel())) - {bg, marker_color}):
        _, components = _label_mask(g == color, connectivity=4)
        for coords in components:
            rr, cc = coords[:, 0], coords[:, 1]
            if int(fg_labels8[int(rr[0]), int(cc[0])]) == marker_component:
                continue
            out[rr, cc] = bg
            r0, r1 = int(rr.min()), int(rr.max())
            c0, c1 = int(cc.min()), int(cc.max())
            out[r0, c0 : c1 + 1] = color
            out[r1, c0 : c1 + 1] = color
            out[r0 : r1 + 1, c0] = color
            out[r0 : r1 + 1, c1] = color

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _grow_shape(box_h: int, box_w: int, area: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    frontier = set(_neighbors(start[0], start[1], box_h, box_w, connectivity=4))
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors(cell[0], cell[1], box_h, box_w, connectivity=4):
            if nbr not in cells:
                frontier.add(nbr)
    if len(cells) != area:
        return None
    return _normalize_cells(cells)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.35:
        out = out.T
    return np.asarray(out, dtype=bool)


def _make_blob(n_regions: int) -> set[tuple[int, int]] | None:
    min_area = n_regions * 5
    for _ in range(80):
        box_h = random.randint(4, 7)
        box_w = random.randint(4, 7)
        max_area = min(box_h * box_w, n_regions * 8 + 3)
        if max_area < min_area:
            continue
        area = random.randint(min_area, max_area)
        cells = _grow_shape(box_h, box_w, area)
        if cells is not None:
            return cells
    return None


def _partition_blob(blob: set[tuple[int, int]], n_regions: int, min_size: int = 3) -> list[set[tuple[int, int]]] | None:
    cells = list(blob)
    if len(cells) < n_regions * min_size:
        return None

    for _ in range(120):
        seeds = random.sample(cells, n_regions)
        regions = [set([seed]) for seed in seeds]
        owner = {seed: idx for idx, seed in enumerate(seeds)}
        frontiers = []
        for seed in seeds:
            frontier = set()
            for nbr in _neighbors(seed[0], seed[1], 1000, 1000, connectivity=4):
                if nbr in blob and nbr not in owner:
                    frontier.add(nbr)
            frontiers.append(frontier)

        while len(owner) < len(cells):
            sizes = [len(region) for region in regions]
            min_region = min(sizes)
            choices: list[tuple[int, tuple[int, int]]] = []
            for idx, frontier in enumerate(frontiers):
                weight = 3 if len(regions[idx]) == min_region else 1
                for cell in frontier:
                    choices.extend([(idx, cell)] * weight)
            if not choices:
                break

            idx, cell = random.choice(choices)
            if cell in owner:
                frontiers[idx].discard(cell)
                continue

            owner[cell] = idx
            regions[idx].add(cell)
            for frontier in frontiers:
                frontier.discard(cell)
            for nbr in _neighbors(cell[0], cell[1], 1000, 1000, connectivity=4):
                if nbr in blob and nbr not in owner:
                    frontiers[idx].add(nbr)

        if len(owner) != len(cells):
            continue
        if min(len(region) for region in regions) < min_size:
            continue
        return regions

    return None


def _make_connected_patch(colors: list[int], marker_color: int) -> np.ndarray | None:
    n_regions = len(colors)
    for _ in range(120):
        blob = _make_blob(n_regions)
        if blob is None:
            continue
        regions = _partition_blob(blob, n_regions)
        if regions is None:
            continue

        mask = _cells_to_mask(blob)
        mask = _random_transform(mask)
        blob = {tuple(map(int, cell)) for cell in np.argwhere(mask)}
        regions = _partition_blob(blob, n_regions)
        if regions is None:
            continue

        local = np.full((mask.shape[0] + 2, mask.shape[1] + 2), BG, dtype=int)
        shifted_regions = []
        for region in regions:
            shifted_regions.append({(r + 1, c + 1) for r, c in region})

        for color, region in zip(colors, shifted_regions):
            for r, c in region:
                local[r, c] = color

        marker_candidates = []
        h, w = local.shape
        for r in range(h):
            for c in range(w):
                if local[r, c] != BG:
                    continue
                if r not in (0, h - 1) and c not in (0, w - 1):
                    continue
                if any(local[nr, nc] != BG for nr, nc in _neighbors(r, c, h, w, connectivity=4)):
                    marker_candidates.append((r, c))
        if not marker_candidates:
            continue

        mr, mc = random.choice(marker_candidates)
        local[mr, mc] = marker_color

        rows, cols = np.argwhere(local != BG).T
        cropped = local[int(rows.min()) : int(rows.max()) + 1, int(cols.min()) : int(cols.max()) + 1]
        return cropped

    return None


def _make_isolated_patch(color: int) -> np.ndarray | None:
    for _ in range(120):
        box_h = random.randint(2, 5)
        box_w = random.randint(2, 5)
        max_area = min(box_h * box_w, 9)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        cells = _grow_shape(box_h, box_w, area)
        if cells is None:
            continue

        mask = _random_transform(_cells_to_mask(cells))
        h, w = mask.shape
        if h < 2 or w < 2:
            continue

        frame = np.zeros_like(mask, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if np.array_equal(mask, frame):
            continue

        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = color
        return patch

    return None


def _can_place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int, gap: int = 1) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False

    for r, c in np.argwhere(patch != BG):
        rr, cc = top + int(r), left + int(c)
        if occupied[rr, cc]:
            return False
        r0 = max(0, rr - gap)
        r1 = min(h, rr + gap + 1)
        c0 = max(0, cc - gap)
        c1 = min(w, cc + gap + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _paste_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for r, c in np.argwhere(patch != BG):
        rr, cc = top + int(r), left + int(c)
        grid[rr, cc] = int(patch[r, c])
        occupied[rr, cc] = True


def _place_random_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for top, left in positions[:180]:
        if _can_place_patch(grid, occupied, patch, top, left, gap=1):
            _paste_patch(grid, occupied, patch, top, left)
            return True
    return False


def _foreground_object_count(grid: np.ndarray, marker_color: int) -> int:
    count = 0
    for color in set(map(int, grid.ravel())) - {BG, marker_color}:
        _, comps = _label_mask(grid == color, connectivity=4)
        count += len(comps)
    return count


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 21)
    w = random.randint(12, 21)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_connected = random.randint(1, 3)
    n_isolated = random.randint(1, 3)
    needed_colors = 1 + n_connected + n_isolated
    palette = list(range(1, 10))
    random.shuffle(palette)
    marker_color = int(palette[0])
    connected_colors = [int(v) for v in palette[1 : 1 + n_connected]]
    isolated_colors = [int(v) for v in palette[1 + n_connected : needed_colors]]

    connected_patch = _make_connected_patch(connected_colors, marker_color)
    if connected_patch is None:
        return None
    if not _place_random_patch(grid, occupied, connected_patch):
        return None

    for color in isolated_colors:
        patch = _make_isolated_patch(color)
        if patch is None:
            return None
        if not _place_random_patch(grid, occupied, patch):
            return None

    counts = Counter(map(int, grid.ravel()))
    if counts[BG] <= max((n for color, n in counts.items() if color != BG), default=0):
        return None
    if counts[marker_color] != 1:
        return None
    if any(counts[color] <= 1 for color in connected_colors + isolated_colors):
        return None
    if _foreground_object_count(grid, marker_color) != n_connected + n_isolated:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
