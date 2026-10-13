import random
from collections import deque

import numpy as np


WALL = 6
FLOOR = 7
# The puzzle text lists pink in the cycle, but ARC pink shares the wall color.
# Use the non-wall colors in the same visible order.
PALETTE = (3, 4, 0, 5, 2, 8, 9, 1)
PALETTE_INDEX = {color: idx for idx, color in enumerate(PALETTE)}
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 300


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(cells)
    return components


def bfs_distances(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(mask.shape, -1, dtype=int)
    sr, sc = start
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def farthest_endpoint(mask: np.ndarray) -> tuple[tuple[int, int], int]:
    cells = np.argwhere(mask)
    start = tuple(int(v) for v in cells[0])
    dist_a = bfs_distances(mask, start)
    a = tuple(int(v) for v in np.argwhere(dist_a == dist_a.max())[0])
    dist_b = bfs_distances(mask, a)
    b = tuple(int(v) for v in np.argwhere(dist_b == dist_b.max())[0])
    return b, int(dist_b[b])


def is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(int(v) for v in cells[0])
    return int((bfs_distances(mask, start) >= 0).sum()) == int(mask.sum())


def rigid_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if in_bounds(mask.shape, nr, nc):
                    out[nr, nc] = True
    return out


def make_simple_room() -> tuple[np.ndarray, bool]:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    return np.ones((h, w), dtype=bool), False


def make_barrier_room() -> tuple[np.ndarray, bool]:
    h = random.randint(6, 10)
    w = random.randint(6, 10)
    mask = np.ones((h, w), dtype=bool)
    thickness = 1 if random.random() < 0.75 else 2
    if random.random() < 0.5:
        col = random.randint(2, w - thickness - 2)
        if random.random() < 0.5:
            end = random.randint(h // 2, h - 2)
            mask[0 : end + 1, col : col + thickness] = False
        else:
            start = random.randint(1, h // 2)
            mask[start:h, col : col + thickness] = False
    else:
        row = random.randint(2, h - thickness - 2)
        if random.random() < 0.5:
            end = random.randint(w // 2, w - 2)
            mask[row : row + thickness, 0 : end + 1] = False
        else:
            start = random.randint(1, w // 2)
            mask[row : row + thickness, start:w] = False
    mask = rigid_transform(mask)
    return mask, True


def make_offset_barbell() -> tuple[np.ndarray, bool]:
    room1_h = random.randint(3, 5)
    room1_w = random.randint(3, 5)
    room2_h = random.randint(3, 5)
    room2_w = random.randint(3, 5)
    width = 1 if random.random() < 0.8 else 2
    gap = random.randint(3, 7)
    offset = random.randint(-3, 3)

    top1 = max(0, -offset)
    top2 = max(0, offset)
    room2_c0 = room1_w + gap

    h = max(top1 + room1_h, top2 + room2_h)
    w = room2_c0 + room2_w
    mask = np.zeros((h, w), dtype=bool)
    mask[top1 : top1 + room1_h, 0:room1_w] = True
    mask[top2 : top2 + room2_h, room2_c0 : room2_c0 + room2_w] = True

    band1 = random.randint(top1, top1 + room1_h - width)
    band2 = random.randint(top2, top2 + room2_h - width)
    mask[band1 : band1 + width, room1_w - 1 : room2_c0 + width] = True
    v0 = min(band1, band2)
    v1 = max(band1 + width, band2 + width)
    mask[v0:v1, room2_c0 : room2_c0 + width] = True
    mask[band2 : band2 + width, room2_c0 : room2_c0 + room2_w] = True

    if random.random() < 0.35:
        spur_len = random.randint(2, 4)
        spur_side = random.choice(["up", "down"])
        spur_col = random.randint(1, w - 2)
        base_row = random.randint(1, h - 2)
        if spur_side == "up":
            r0 = max(0, base_row - spur_len)
            mask[r0 : base_row + 1, spur_col] = True
        else:
            r1 = min(h, base_row + spur_len + 1)
            mask[base_row:r1, spur_col] = True

    mask = rigid_transform(mask)
    return mask, True


def make_region() -> tuple[np.ndarray, bool]:
    maker = random.choices(
        [make_simple_room, make_barrier_room, make_offset_barbell],
        weights=[2, 3, 4],
        k=1,
    )[0]
    for _ in range(20):
        mask, complex_region = maker()
        if is_connected(mask):
            return mask, complex_region
    raise RuntimeError("failed to create a connected region")


def choose_marker(mask: np.ndarray, prefer_long: bool) -> tuple[int, int]:
    if prefer_long:
        endpoint, _diameter = farthest_endpoint(mask)
        return endpoint

    cells = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    roomy = []
    for r, c in cells:
        degree = sum(
            1
            for nr, nc in neighbors4(r, c)
            if in_bounds(mask.shape, nr, nc) and mask[nr, nc]
        )
        if degree >= 2:
            roomy.append((r, c))
    return random.choice(roomy or cells)


def can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    candidate = np.zeros_like(occupied, dtype=bool)
    h, w = mask.shape
    candidate[top : top + h, left : left + w] = mask
    return not np.any(dilate8(candidate) & occupied)


def gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 28)
    w = random.randint(15, 28)
    grid = np.full((h, w), WALL, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    target_regions = random.randint(2, 5)
    regions = []
    for region_idx in range(target_regions):
        placed = False
        for _ in range(80):
            if region_idx == 0:
                mask, complex_region = random.choice([make_barrier_room, make_offset_barbell])()
            else:
                mask, complex_region = make_region()
            mh, mw = mask.shape
            if mh > h or mw > w:
                continue
            top = random.randint(0, h - mh)
            left = random.randint(0, w - mw)
            if not can_place(occupied, mask, top, left):
                continue
            occupied[top : top + mh, left : left + mw] |= mask
            grid[top : top + mh, left : left + mw][mask] = FLOOR
            endpoint, diameter = farthest_endpoint(mask)
            regions.append(
                {
                    "mask": mask,
                    "top": top,
                    "left": left,
                    "complex": complex_region,
                    "diameter": diameter,
                    "endpoint": endpoint,
                }
            )
            placed = True
            break
        if not placed and region_idx < 2:
            return None

    if len(regions) < 2:
        return None

    complex_ids = [idx for idx, region in enumerate(regions) if region["complex"]]
    simple_ids = [idx for idx, region in enumerate(regions) if not region["complex"]]
    if not complex_ids:
        return None

    mark_count = random.randint(1, len(regions))
    if len(regions) > 1 and mark_count == len(regions) and random.random() < 0.75:
        mark_count -= 1

    marked: set[int] = {random.choice(complex_ids)}
    if simple_ids and mark_count > 1 and random.random() < 0.7:
        marked.add(random.choice(simple_ids))
    while len(marked) < mark_count:
        marked.add(random.randrange(len(regions)))

    if all(regions[idx]["diameter"] <= len(PALETTE) for idx in marked) and random.random() < 0.6:
        return None

    colors = random.sample(PALETTE, len(marked))
    for color, idx in zip(colors, marked):
        region = regions[idx]
        prefer_long = bool(region["complex"] or random.random() < 0.4)
        local_r, local_c = choose_marker(region["mask"], prefer_long=prefer_long)
        grid[region["top"] + local_r, region["left"] + local_c] = color

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each marker-bearing floor region into a cyclic geodesic wavefront around its marker.

    Input structure:
    - The canvas consists of orange floor and magenta wall cells; walls block 4-neighbor movement and split the floor into separate 4-connected regions.
    - Some regions contain exactly one non-orange, non-wall marker pixel.
    - Marker colors follow the puzzle's fixed visible cycle; in ARC ids the usable non-wall order is green-yellow-black-grey-red-teal-maroon-blue.

    Rule:
    - Keep every wall cell unchanged.
    - For each region that contains a marker, compute shortest 4-neighbor path lengths from that marker while staying inside the same floor region.
    - Recolor the marker cell and every other cell in that region by advancing through the cyclic palette by the geodesic distance; leave unmarked floor regions orange.

    Key insight:
    - The wavefront uses distance around walls rather than raw Manhattan distance, so equal-color bands bend through corridors and shift across thin barriers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    traversable = g != WALL
    out = g.copy()

    for cells in component_cells(traversable):
        markers = [(r, c) for r, c in cells if int(g[r, c]) != FLOOR]
        if not markers:
            continue

        assert len(markers) == 1, "each floor region must contain at most one marker"
        sr, sc = markers[0]
        start_color = int(g[sr, sc])
        assert start_color in PALETTE_INDEX, "marker color must belong to the cyclic palette"

        comp_mask = np.zeros_like(traversable, dtype=bool)
        for r, c in cells:
            comp_mask[r, c] = True
        dist = bfs_distances(comp_mask, (sr, sc))
        base_idx = PALETTE_INDEX[start_color]
        for r, c in cells:
            out[r, c] = PALETTE[(base_idx + int(dist[r, c])) % len(PALETTE)]

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wavefront-coloring input")
