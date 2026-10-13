from collections import deque
from itertools import permutations

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

_RNG = np.random.default_rng()
GEN_RETRIES = 300
PATCH_RETRIES = 120
MIN_ROOM_SIZE = 2


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    dirs = DIR4 if connectivity == 4 else DIR8
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
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _signed_area(order: list[tuple[int, int]]) -> int:
    total = 0
    for i, (r1, c1) in enumerate(order):
        r2, c2 = order[(i + 1) % len(order)]
        total += c1 * r2 - c2 * r1
    return int(total)


def _trace_clockwise_cycle(coords: np.ndarray) -> list[tuple[int, int]]:
    points = {tuple(map(int, cell)) for cell in coords}
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in points:
        here = []
        for dr, dc in DIR8:
            nxt = (r + dr, c + dc)
            if nxt in points:
                here.append(nxt)
        if len(here) != 2:
            raise ValueError("ring is not a simple single-pixel cycle")
        neighbors[(r, c)] = here

    start = min(points)
    seq = [start]
    prev = start
    cur = neighbors[start][0]
    while cur != start:
        if len(seq) > len(points):
            raise ValueError("cycle trace overflow")
        seq.append(cur)
        nxt = [pt for pt in neighbors[cur] if pt != prev]
        if len(nxt) != 1:
            raise ValueError("cycle branch detected")
        prev, cur = cur, nxt[0]

    if len(seq) != len(points):
        raise ValueError("cycle trace incomplete")
    if _signed_area(seq) < 0:
        seq = [seq[0]] + list(reversed(seq[1:]))
    return seq


def _find_ring_and_containers(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    mask = (grid != BLACK) & (grid != GREY)
    comps = _components(mask, connectivity=8)
    ring_candidates: list[np.ndarray] = []
    containers: list[tuple[int, np.ndarray]] = []

    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) > 1:
            ring_candidates.append(coords)
        elif colors:
            containers.append((int(next(iter(colors))), coords))

    if len(ring_candidates) != 1:
        raise ValueError("expected exactly one multicolor ring")
    if not containers:
        raise ValueError("expected at least one container")
    return ring_candidates[0], containers


def _count_chambers(grid: np.ndarray, coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    blacks = _components(patch == BLACK, connectivity=4)
    h, w = patch.shape
    count = 0
    for comp in blacks:
        if (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == h - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == w - 1)
        ):
            continue
        count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each container by the clockwise ring color indexed by its chamber count.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    ring_coords, containers = _find_ring_and_containers(g)
    ring_order = _trace_clockwise_cycle(ring_coords)
    ring_colors = [int(g[r, c]) for r, c in ring_order]

    for _color, coords in containers:
        chambers = _count_chambers(g, coords)
        if not 1 <= chambers <= len(ring_colors):
            raise ValueError("chamber count exceeds ring length")
        out[coords[:, 0], coords[:, 1]] = ring_colors[chambers - 1]

    return out


def _inner_size_range(chambers: int) -> tuple[int, int]:
    if chambers <= 2:
        return 5, 7
    if chambers <= 4:
        return 6, 8
    return 8, 9


def _build_partition(rng: np.random.Generator, inner_h: int, inner_w: int, chambers: int) -> np.ndarray | None:
    inner = np.zeros((inner_h, inner_w), dtype=int)
    rooms = [(0, 0, inner_h - 1, inner_w - 1)]

    for _ in range(chambers - 1):
        candidates: list[tuple[int, str]] = []
        for idx, (r0, c0, r1, c1) in enumerate(rooms):
            if (r1 - r0 + 1) >= 2 * MIN_ROOM_SIZE + 1:
                candidates.append((idx, "h"))
            if (c1 - c0 + 1) >= 2 * MIN_ROOM_SIZE + 1:
                candidates.append((idx, "v"))
        if not candidates:
            return None

        room_idx, axis = candidates[int(rng.integers(len(candidates)))]
        r0, c0, r1, c1 = rooms.pop(room_idx)

        if axis == "h":
            wall = int(rng.integers(r0 + MIN_ROOM_SIZE, r1 - MIN_ROOM_SIZE + 1))
            inner[wall, c0 : c1 + 1] = GREY
            rooms.append((r0, c0, wall - 1, c1))
            rooms.append((wall + 1, c0, r1, c1))
        else:
            wall = int(rng.integers(c0 + MIN_ROOM_SIZE, c1 - MIN_ROOM_SIZE + 1))
            inner[r0 : r1 + 1, wall] = GREY
            rooms.append((r0, c0, r1, wall - 1))
            rooms.append((r0, wall + 1, r1, c1))

    return inner


def _crop_nonzero(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _add_bumps(rng: np.random.Generator, base: np.ndarray, color: int) -> np.ndarray:
    pad = 1
    h, w = base.shape
    work = np.zeros((h + 2 * pad, w + 2 * pad), dtype=int)
    work[pad : pad + h, pad : pad + w] = base

    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[: int(rng.integers(0, 3))]:
        if side in {"top", "bottom"} and w >= 4:
            span = int(rng.integers(2, w))
            start = int(rng.integers(0, w - span + 1))
            cols = slice(pad + start, pad + start + span)
            if side == "top":
                work[:pad, cols] = color
            else:
                work[pad + h :, cols] = color
        if side in {"left", "right"} and h >= 4:
            span = int(rng.integers(2, h))
            start = int(rng.integers(0, h - span + 1))
            rows = slice(pad + start, pad + start + span)
            if side == "left":
                work[rows, :pad] = color
            else:
                work[rows, pad + w :] = color

    return _crop_nonzero(work)


def _make_container_patch(rng: np.random.Generator, color: int, chambers: int) -> np.ndarray | None:
    size_lo, size_hi = _inner_size_range(chambers)
    for _ in range(PATCH_RETRIES):
        inner_h = int(rng.integers(size_lo, size_hi + 1))
        inner_w = int(rng.integers(size_lo, size_hi + 1))
        inner = _build_partition(rng, inner_h, inner_w, chambers)
        if inner is None:
            continue

        base = np.full((inner_h + 2, inner_w + 2), color, dtype=int)
        base[1:-1, 1:-1] = inner
        patch = _add_bumps(rng, base, color)

        color_coords = np.argwhere(patch == color)
        if color_coords.size == 0:
            continue
        if _count_chambers(patch, color_coords) != chambers:
            continue
        return patch
    return None


def _diamond_ring_mask(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    center = radius
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        dx = radius - abs(r - center)
        if dx == 0:
            mask[r, center] = True
        else:
            mask[r, center - dx] = True
            mask[r, center + dx] = True
    return mask


def _make_ring_sequence(rng: np.random.Generator, colors: list[int], length: int) -> list[int]:
    distinct = int(rng.integers(3, len(colors) + 1))
    base = list(map(int, rng.choice(colors, size=distinct, replace=False)))
    seq: list[int] = []
    while len(seq) < length:
        chunk = base.copy()
        rng.shuffle(chunk)
        if seq and chunk[0] == seq[-1]:
            chunk = chunk[1:] + chunk[:1]
        seq.extend(chunk)
    seq = seq[:length]
    if seq[0] == seq[-1]:
        for color in base:
            if color != seq[0]:
                seq[-1] = color
                break
    return seq


def _make_ring_patch(rng: np.random.Generator, colors: list[int], radius: int) -> np.ndarray:
    mask = _diamond_ring_mask(radius)
    coords = np.argwhere(mask)
    order = _trace_clockwise_cycle(coords)
    patch = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(order, colors):
        patch[r, c] = int(color)
    return patch


def _fits(patch: np.ndarray, zone: tuple[int, int, int, int]) -> bool:
    h, w = patch.shape
    r0, r1, c0, c1 = zone
    return h <= (r1 - r0) and w <= (c1 - c0)


def _zone_position(
    rng: np.random.Generator,
    patch: np.ndarray,
    zone: tuple[int, int, int, int],
) -> tuple[int, int]:
    h, w = patch.shape
    r0, r1, c0, c1 = zone
    top = int(rng.integers(r0, r1 - h + 1))
    left = int(rng.integers(c0, c1 - w + 1))
    return top, left


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[top : top + h, left : left + w] = patch


def _validate_scene(grid: np.ndarray, expected_chambers: dict[int, int]) -> bool:
    try:
        ring_coords, containers = _find_ring_and_containers(grid)
        if len(containers) != len(expected_chambers):
            return False

        top_row = int(ring_coords[:, 0].min())
        if int(np.sum(ring_coords[:, 0] == top_row)) != 1:
            return False

        ring_order = _trace_clockwise_cycle(ring_coords)
        ring_colors = [int(grid[r, c]) for r, c in ring_order]
        actual_chambers = {int(color): _count_chambers(grid, coords) for color, coords in containers}
        if actual_chambers != expected_chambers:
            return False

        out = main(grid)
        if not np.array_equal(out[grid == GREY], grid[grid == GREY]):
            return False
        if not np.array_equal(out[ring_coords[:, 0], ring_coords[:, 1]], grid[ring_coords[:, 0], ring_coords[:, 1]]):
            return False

        for color, coords in containers:
            new_color = ring_colors[actual_chambers[int(color)] - 1]
            if not np.all(out[coords[:, 0], coords[:, 1]] == new_color):
                return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    container_count = int(rng.integers(2, 4))
    color_order = COLOR_POOL.copy()
    rng.shuffle(color_order)
    container_colors = color_order[:container_count]
    ring_pool = color_order[container_count:]
    if len(ring_pool) < 3:
        return None

    room_counts = list(map(int, rng.choice(np.arange(1, 7), size=container_count, replace=False)))
    rng.shuffle(room_counts)

    container_specs: list[tuple[np.ndarray, int, int]] = []
    for color, chambers in zip(container_colors, room_counts):
        patch = _make_container_patch(rng, int(color), int(chambers))
        if patch is None:
            return None
        container_specs.append((patch, int(color), int(chambers)))

    max_chambers = max(room_counts)
    min_radius = max(2, (max_chambers + 3) // 4)
    radius = int(rng.integers(min_radius, 4))
    ring_length = 4 * radius
    ring_colors = _make_ring_sequence(rng, ring_pool, ring_length)
    ring_patch = _make_ring_patch(rng, ring_colors, radius)

    h = int(rng.integers(28, 31))
    w = int(rng.integers(28, 31))
    mid_r = h // 2
    mid_c = w // 2

    ring_zone = (1, mid_r, mid_c + 1, w - 1)
    container_zones = [
        (1, mid_r, 1, mid_c),
        (mid_r + 1, h - 1, 1, mid_c),
        (mid_r + 1, h - 1, mid_c + 1, w - 1),
    ]

    if not _fits(ring_patch, ring_zone):
        return None

    zone_orders = list(permutations(range(len(container_zones)), len(container_specs)))
    rng.shuffle(zone_orders)
    zone_assignment = None
    ordered_specs = sorted(container_specs, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True)
    for zone_ids in zone_orders:
        if all(_fits(spec[0], container_zones[zone_id]) for spec, zone_id in zip(ordered_specs, zone_ids)):
            zone_assignment = list(zip(ordered_specs, zone_ids))
            break
    if zone_assignment is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    ring_top, ring_left = _zone_position(rng, ring_patch, ring_zone)
    _paste(grid, ring_patch, ring_top, ring_left)

    expected_chambers: dict[int, int] = {}
    for (patch, color, chambers), zone_id in zone_assignment:
        top, left = _zone_position(rng, patch, container_zones[zone_id])
        _paste(grid, patch, top, left)
        expected_chambers[int(color)] = int(chambers)

    if not _validate_scene(grid, expected_chambers):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate containers with grey-partitioned black chambers plus a clockwise color ring.
    """
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chamber-ring puzzle")
