from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
PALETTE_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATE_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _line_segments(line: np.ndarray) -> list[int]:
    values = [int(x) for x in line.tolist()]
    segments = [values[0]]
    for value in values[1:]:
        if value != segments[-1]:
            segments.append(value)
    return segments


def _find_palette(grid: np.ndarray) -> tuple[str, list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[str, list[int], np.ndarray]] = []
    edges = [
        ("top", grid[0, :]),
        ("bottom", grid[h - 1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, w - 1]),
    ]
    for side, line in edges:
        values = [int(x) for x in line.tolist()]
        if any(value in (BG, WALL) for value in values):
            continue
        palette = _line_segments(line)
        if not (3 <= len(palette) <= 5):
            continue
        if len(set(palette)) != len(palette):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        if side == "top":
            mask[0, :] = True
        elif side == "bottom":
            mask[h - 1, :] = True
        elif side == "left":
            mask[:, 0] = True
        else:
            mask[:, w - 1] = True
        candidates.append((side, palette, mask))
    assert len(candidates) == 1, "expected exactly one palette strip"
    return candidates[0]


def _connected_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_boundary = False
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                touches_boundary |= rr in (0, h - 1) or cc in (0, w - 1)
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            components.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "touches_boundary": touches_boundary,
                }
            )
    return components


def _find_regions(grid: np.ndarray, palette_mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    traversable = (grid != WALL) & (~palette_mask)
    label = np.zeros(grid.shape, dtype=int)
    regions: list[tuple[int, int, int, int]] = []
    next_id = 1
    for comp in _connected_components(traversable):
        if comp["touches_boundary"]:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        area = len(comp["coords"])
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "expected rectangular partitions"
        for r, c in comp["coords"]:
            label[r, c] = next_id
        regions.append((r0, c0, r1, c1))
        next_id += 1
    assert regions, "expected enclosed partitions"
    return label, regions


def _find_seed(grid: np.ndarray, label: np.ndarray, n_regions: int) -> tuple[int, np.ndarray]:
    seed_candidates: list[tuple[int, np.ndarray]] = []
    for rid in range(1, n_regions + 1):
        coords = np.argwhere((label == rid) & (grid != BG))
        if coords.size:
            seed_candidates.append((rid, coords))
    assert len(seed_candidates) == 1, "expected a unique seed partition"
    seed_rid, coords = seed_candidates[0]
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    pattern = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] != BG
    assert bool(pattern.any()), "seed pattern must be non-empty"
    return int(seed_rid), pattern


def _adjacency_from_walls(grid: np.ndarray, label: np.ndarray, n_regions: int) -> list[set[int]]:
    h, w = grid.shape
    neighbors = [set() for _ in range(n_regions + 1)]
    for r, c in np.argwhere(grid == WALL):
        r = int(r)
        c = int(c)
        if 0 < c < w - 1:
            left = int(label[r, c - 1])
            right = int(label[r, c + 1])
            if left > 0 and right > 0 and left != right:
                neighbors[left].add(right)
                neighbors[right].add(left)
        if 0 < r < h - 1:
            up = int(label[r - 1, c])
            down = int(label[r + 1, c])
            if up > 0 and down > 0 and up != down:
                neighbors[up].add(down)
                neighbors[down].add(up)
    return neighbors


def _bfs_layers(neighbors: list[set[int]], seed_rid: int) -> list[int]:
    layers = [-1] * len(neighbors)
    q = deque([seed_rid])
    layers[seed_rid] = 0
    while q:
        node = q.popleft()
        for nxt in neighbors[node]:
            if layers[nxt] == -1:
                layers[nxt] = layers[node] + 1
                q.append(nxt)
    assert all(layer >= 0 for layer in layers[1:]), "all partitions should be reachable"
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a centered seed pattern through walled partitions, cycling colors with palette-strip distance.
    """
    g = np.asarray(grid, dtype=int)
    palette_side, palette, palette_mask = _find_palette(g)
    _ = palette_side  # The side is only structural evidence; color order drives the transformation.

    label, regions = _find_regions(g, palette_mask)
    n_regions = len(regions)
    seed_rid, pattern = _find_seed(g, label, n_regions)
    ph, pw = pattern.shape

    neighbors = _adjacency_from_walls(g, label, n_regions)
    layers = _bfs_layers(neighbors, seed_rid)

    out = g.copy()
    for rid in range(1, n_regions + 1):
        out[label == rid] = BG

    for rid, (r0, c0, r1, c1) in enumerate(regions, start=1):
        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1
        if room_h < ph or room_w < pw:
            continue
        sr = r0 + (room_h - ph) // 2
        sc = c0 + (room_w - pw) // 2
        room_slice = label[sr : sr + ph, sc : sc + pw]
        if not np.all(room_slice[pattern] == rid):
            continue
        color = palette[layers[rid] % len(palette)]
        patch = out[sr : sr + ph, sc : sc + pw]
        patch[pattern] = int(color)
    return out


def _random_pattern() -> np.ndarray:
    for _ in range(100):
        h = random.randint(1, 4)
        w = random.randint(1, 4)
        if h == 1 and w == 1:
            continue
        target = random.randint(2, min(h * w, 6))
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        cells = [start]
        while int(mask.sum()) < target:
            frontier = []
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            choice = random.choice(frontier)
            if not mask[choice]:
                mask[choice] = True
                cells.append(choice)
        if int(mask.sum()) < 2:
            continue
        coords = np.argwhere(mask)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        if random.random() < 0.75 and bool(cropped.all()) and cropped.size > 2:
            continue
        return cropped
    raise RuntimeError("failed to sample a seed pattern")


def _sample_region_sizes(count: int, need: int, force_small: bool) -> list[int]:
    sizes: list[int] = []
    for _ in range(count):
        if need > 1 and random.random() < 0.35:
            sizes.append(random.randint(1, need - 1))
        else:
            sizes.append(random.randint(need, need + 3))
    if all(size < need for size in sizes):
        sizes[random.randrange(count)] = random.randint(need, need + 3)
    if force_small and need > 1 and all(size >= need for size in sizes):
        idx = random.randrange(count)
        sizes[idx] = random.randint(1, need - 1)
        if all(size < need for size in sizes):
            sizes[(idx + 1) % count] = random.randint(need, need + 3)
    return sizes


def _room_bboxes(net_r0: int, net_c0: int, row_heights: list[int], col_widths: list[int]) -> list[list[tuple[int, int, int, int]]]:
    rooms: list[list[tuple[int, int, int, int]]] = []
    r = net_r0 + 1
    for height in row_heights:
        row_rooms: list[tuple[int, int, int, int]] = []
        c = net_c0 + 1
        for width in col_widths:
            row_rooms.append((r, c, r + height - 1, c + width - 1))
            c += width + 1
        rooms.append(row_rooms)
        r += height + 1
    return rooms


def _draw_partition_walls(grid: np.ndarray, net_r0: int, net_c0: int, row_heights: list[int], col_widths: list[int]) -> None:
    net_h = sum(row_heights) + len(row_heights) + 1
    net_w = sum(col_widths) + len(col_widths) + 1
    grid[net_r0, net_c0 : net_c0 + net_w] = WALL
    grid[net_r0 + net_h - 1, net_c0 : net_c0 + net_w] = WALL
    grid[net_r0 : net_r0 + net_h, net_c0] = WALL
    grid[net_r0 : net_r0 + net_h, net_c0 + net_w - 1] = WALL

    r = net_r0
    for height in row_heights[:-1]:
        r += height + 1
        grid[r, net_c0 : net_c0 + net_w] = WALL
    c = net_c0
    for width in col_widths[:-1]:
        c += width + 1
        grid[net_r0 : net_r0 + net_h, c] = WALL


def _stamp_centered_pattern(grid: np.ndarray, bbox: tuple[int, int, int, int], pattern: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = bbox
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1
    ph, pw = pattern.shape
    sr = r0 + (room_h - ph) // 2
    sc = c0 + (room_w - pw) // 2
    patch = grid[sr : sr + ph, sc : sc + pw]
    patch[pattern] = int(color)


def _random_palette_line(length: int, palette: list[int]) -> np.ndarray | None:
    if length < len(palette):
        return None
    cuts = sorted(random.sample(range(1, length), len(palette) - 1))
    parts = np.diff(np.array([0, *cuts, length]))
    line = np.zeros(length, dtype=int)
    start = 0
    for color, span in zip(palette, parts):
        line[start : start + int(span)] = int(color)
        start += int(span)
    return line


def _expected_output(
    inp: np.ndarray,
    rooms: list[list[tuple[int, int, int, int]]],
    seed_pos: tuple[int, int],
    pattern: np.ndarray,
    palette: list[int],
) -> np.ndarray:
    out = inp.copy()
    rows = len(rooms)
    cols = len(rooms[0])
    for i in range(rows):
        for j in range(cols):
            r0, c0, r1, c1 = rooms[i][j]
            out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for i in range(rows):
        for j in range(cols):
            bbox = rooms[i][j]
            r0, c0, r1, c1 = bbox
            room_h = r1 - r0 + 1
            room_w = c1 - c0 + 1
            ph, pw = pattern.shape
            if room_h < ph or room_w < pw:
                continue
            layer = abs(i - seed_pos[0]) + abs(j - seed_pos[1])
            _stamp_centered_pattern(out, bbox, pattern, palette[layer % len(palette)])
    return out


def _gen_attempt() -> np.ndarray | None:
    pattern = _random_pattern()
    ph, pw = pattern.shape

    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    force_small = random.random() < 0.5

    row_heights = _sample_region_sizes(n_rows, ph, force_small)
    col_widths = _sample_region_sizes(n_cols, pw, force_small)

    fit_rooms = [
        (i, j)
        for i in range(n_rows)
        for j in range(n_cols)
        if row_heights[i] >= ph and col_widths[j] >= pw
    ]
    if len(fit_rooms) < 2:
        return None
    if force_small and not any(
        row_heights[i] < ph or col_widths[j] < pw
        for i in range(n_rows)
        for j in range(n_cols)
    ):
        return None

    corners = {(0, 0), (0, n_cols - 1), (n_rows - 1, 0), (n_rows - 1, n_cols - 1)}
    corner_fits = [room for room in fit_rooms if room in corners]
    if corner_fits and random.random() < 0.5:
        seed_pos = random.choice(corner_fits)
    else:
        seed_pos = random.choice(fit_rooms)

    palette_len = random.randint(3, 5)
    palette = random.sample(PALETTE_POOL, palette_len)
    max_layer = max(abs(i - seed_pos[0]) + abs(j - seed_pos[1]) for i in range(n_rows) for j in range(n_cols))
    if random.random() < 0.5 and max_layer < palette_len:
        return None

    side = random.choice(["top", "bottom", "left", "right"])
    net_h = sum(row_heights) + n_rows + 1
    net_w = sum(col_widths) + n_cols + 1
    top_extra = 2 if side == "top" else 1
    bottom_extra = 2 if side == "bottom" else 1
    left_extra = 2 if side == "left" else 1
    right_extra = 2 if side == "right" else 1
    h = top_extra + net_h + bottom_extra
    w = left_extra + net_w + right_extra
    if not (10 <= h <= 30 and 10 <= w <= 30):
        return None

    grid = np.full((h, w), BG, dtype=int)
    net_r0 = top_extra
    net_c0 = left_extra
    _draw_partition_walls(grid, net_r0, net_c0, row_heights, col_widths)
    rooms = _room_bboxes(net_r0, net_c0, row_heights, col_widths)
    _stamp_centered_pattern(grid, rooms[seed_pos[0]][seed_pos[1]], pattern, palette[0])

    palette_line = _random_palette_line(w if side in ("top", "bottom") else h, palette)
    if palette_line is None:
        return None
    if side == "top":
        grid[0, :] = palette_line
    elif side == "bottom":
        grid[h - 1, :] = palette_line
    elif side == "left":
        grid[:, 0] = palette_line
    else:
        grid[:, w - 1] = palette_line

    expected = _expected_output(grid, rooms, seed_pos, pattern, palette)
    try:
        result = main(grid)
    except Exception:
        return None
    if np.array_equal(result, grid):
        return None
    if not np.array_equal(result, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
