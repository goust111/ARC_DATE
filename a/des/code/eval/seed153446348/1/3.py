import random
from collections import deque

import numpy as np


BG = 0
OUTLINE = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator config.
GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (1, 4)
PATCH_CANVAS_RANGE = (8, 13)
RECT_MARGIN_RANGE = (2, 3)
RECT_SIDE_RANGE = (4, 8)
BUMP_COUNT_CHOICES = [0, 1, 2, 3, 4]
BUMP_COUNT_WEIGHTS = [1, 3, 3, 2, 1]
PATCH_RETRIES = 200
BUMP_RETRIES = 40
PLACEMENT_PADDING = 2
PLACEMENT_ATTEMPTS = 150
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _cycle_interior_mask(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    path = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        path[r, c] = True

    exterior = np.zeros((h, w), dtype=bool)
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not path[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not path[r, c] and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not path[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))

    return (~path) & (~exterior)


def _trace_cycle(
    start: tuple[int, int],
    nxt: tuple[int, int],
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    seq = [start]
    prev = start
    cur = nxt
    while cur != start:
        seq.append(cur)
        a, b = adjacency[cur]
        cur, prev = (b, cur) if a == prev else (a, cur)
    return seq


def _clockwise_score(
    seq: list[tuple[int, int]],
    interior: np.ndarray,
    origin: tuple[int, int],
) -> int:
    r0, c0 = origin
    h, w = interior.shape

    def inside(rr: int, cc: int) -> int:
        rr -= r0
        cc -= c0
        return int(0 <= rr < h and 0 <= cc < w and interior[rr, cc])

    score = 0
    wrapped = seq[1:] + seq[:1]
    for (r1, c1), (r2, c2) in zip(seq, wrapped):
        dr = r2 - r1
        dc = c2 - c1
        right = (dc, -dr)
        left = (-dc, dr)
        score += inside(r1 + right[0], c1 + right[1])
        score += inside(r2 + right[0], c2 + right[1])
        score -= inside(r1 + left[0], c1 + left[1])
        score -= inside(r2 + left[0], c2 + left[1])
    return score


def _validate_cycle(cycle: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    if len(cycle) < 8 or len(set(cycle)) != len(cycle):
        return False

    coord_set = set(cycle)
    for r, c in cycle:
        if not (0 <= r < h and 0 <= c < w):
            return False
    for (r1, c1), (r2, c2) in zip(cycle, cycle[1:] + cycle[:1]):
        if abs(r2 - r1) + abs(c2 - c1) != 1:
            return False
    for r, c in cycle:
        degree = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))
        if degree != 2:
            return False

    interior = _cycle_interior_mask(cycle, shape)
    if not interior.any():
        return False
    return len(_component_coords(interior)) == 1


def _rectangle_cycle(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    cycle: list[tuple[int, int]] = []
    for c in range(left, left + width):
        cycle.append((top, c))
    for r in range(top + 1, top + height):
        cycle.append((r, left + width - 1))
    for c in range(left + width - 2, left - 1, -1):
        cycle.append((top + height - 1, c))
    for r in range(top + height - 2, top, -1):
        cycle.append((r, left))
    return cycle


def _rotate_list(items: list[tuple[int, int]], start: int) -> list[tuple[int, int]]:
    return items[start:] + items[:start]


def _delta(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return b[0] - a[0], b[1] - a[1]


def _try_bump(cycle: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    n = len(cycle)
    for _ in range(BUMP_RETRIES):
        edge_idx = random.randrange(n)
        direction = _delta(cycle[edge_idx], cycle[(edge_idx + 1) % n])

        run_start = edge_idx
        while _delta(cycle[(run_start - 1) % n], cycle[run_start]) == direction:
            run_start = (run_start - 1) % n
            if run_start == edge_idx:
                break

        rotated = _rotate_list(cycle, run_start)
        direction = _delta(rotated[0], rotated[1])

        run_edges = 1
        while run_edges < len(rotated) - 1 and _delta(rotated[run_edges], rotated[(run_edges + 1) % len(rotated)]) == direction:
            run_edges += 1

        run_cells = run_edges + 1
        if run_cells < 4:
            continue

        a = random.randint(0, run_cells - 4)
        b = random.randint(a + 2, run_cells - 1)
        start = rotated[a]
        end = rotated[b]
        dr, dc = direction
        nr, nc = -dc, dr  # left normal = outward side for clockwise cycle
        span = b - a

        candidates = []
        for depth in (1, 2):
            segment = [start]
            for k in range(1, depth + 1):
                segment.append((start[0] + nr * k, start[1] + nc * k))
            for step in range(1, span + 1):
                segment.append((start[0] + nr * depth + dr * step, start[1] + nc * depth + dc * step))
            for k in range(depth - 1, -1, -1):
                segment.append((end[0] + nr * k, end[1] + nc * k))

            candidate = rotated[:a] + segment + rotated[b + 1 :]
            if _validate_cycle(candidate, shape):
                candidates.append(candidate)

        if candidates:
            return random.choice(candidates)
    return None


def _build_patch(color: int) -> np.ndarray | None:
    for _ in range(PATCH_RETRIES):
        canvas_h = random.randint(*PATCH_CANVAS_RANGE)
        canvas_w = random.randint(*PATCH_CANVAS_RANGE)
        shape = (canvas_h, canvas_w)

        margin_r = random.randint(*RECT_MARGIN_RANGE)
        margin_c = random.randint(*RECT_MARGIN_RANGE)
        max_h = min(RECT_SIDE_RANGE[1], canvas_h - margin_r - 2)
        max_w = min(RECT_SIDE_RANGE[1], canvas_w - margin_c - 2)
        if max_h < RECT_SIDE_RANGE[0] or max_w < RECT_SIDE_RANGE[0]:
            continue

        height = random.randint(RECT_SIDE_RANGE[0], max_h)
        width = random.randint(RECT_SIDE_RANGE[0], max_w)
        cycle = _rectangle_cycle(margin_r, margin_c, height, width)
        if not _validate_cycle(cycle, shape):
            continue

        target_bumps = random.choices(BUMP_COUNT_CHOICES, weights=BUMP_COUNT_WEIGHTS, k=1)[0]
        applied_bumps = 0
        for _ in range(target_bumps):
            bumped = _try_bump(cycle, shape)
            if bumped is not None:
                cycle = bumped
                applied_bumps += 1
        if target_bumps > 0 and applied_bumps == 0:
            continue

        mask = np.zeros(shape, dtype=int)
        for r, c in cycle:
            mask[r, c] = OUTLINE
        r0, c0, r1, c1 = _bbox(cycle)
        patch = mask[r0 : r1 + 1, c0 : c1 + 1]

        patch = np.rot90(patch, random.randrange(4))
        if random.random() < 0.5:
            patch = np.fliplr(patch)

        seed_coords = np.argwhere(patch == OUTLINE)
        if seed_coords.shape[0] < 8:
            continue
        sr, sc = map(int, random.choice(seed_coords.tolist()))
        patch = patch.copy()
        patch[sr, sc] = int(color)
        return patch
    return None


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    patches = sorted(patches, key=lambda p: int(np.count_nonzero(p)), reverse=True)
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    padded_area = sum((int(p.shape[0]) + 2 * PLACEMENT_PADDING) * (int(p.shape[1]) + 2 * PLACEMENT_PADDING) for p in patches)
    base_side = max(GRID_MIN, max_h + 2, max_w + 2, int(np.ceil(np.sqrt(padded_area))) + 1)
    if base_side > GRID_MAX:
        return None

    max_side = min(GRID_MAX, base_side + 5)
    grid_h = random.randint(base_side, max_side)
    grid_w = random.randint(base_side, max_side)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    for patch in patches:
        h, w = map(int, patch.shape)
        if h + 2 > grid_h or w + 2 > grid_w:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r = random.randint(1, grid_h - h - 1)
            c = random.randint(1, grid_w - w - 1)
            r0 = max(0, r - PLACEMENT_PADDING)
            c0 = max(0, c - PLACEMENT_PADDING)
            r1 = min(grid_h, r + h + PLACEMENT_PADDING)
            c1 = min(grid_w, c + w + PLACEMENT_PADDING)
            if reserved[r0:r1, c0:c1].any():
                continue

            mask = patch != BG
            grid[r : r + h, c : c + w][mask] = patch[mask]
            reserved[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fg = g[g != BG]
    assert fg.size > 0
    vals, counts = np.unique(fg, return_counts=True)
    outline = int(vals[counts.argmax()])

    out = g.copy()
    for coords in _component_coords(g != BG):
        coord_set = set(coords)
        seed_cells = [(r, c) for r, c in coords if int(g[r, c]) != outline]
        assert len(seed_cells) == 1
        seed = seed_cells[0]
        seed_color = int(g[seed])

        adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for r, c in coords:
            neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in coord_set]
            assert len(neighbors) == 2
            adjacency[(r, c)] = neighbors

        r0, c0, r1, c1 = _bbox(coords)
        local_coords = [(r - r0, c - c0) for r, c in coords]
        interior = _cycle_interior_mask(local_coords, (r1 - r0 + 1, c1 - c0 + 1))
        assert interior.any()

        first, second = adjacency[seed]
        seq1 = _trace_cycle(seed, first, adjacency)
        seq2 = _trace_cycle(seed, second, adjacency)
        score1 = _clockwise_score(seq1, interior, (r0, c0))
        score2 = _clockwise_score(seq2, interior, (r0, c0))
        assert score1 != score2

        clockwise = seq1 if score1 > score2 else seq2
        for r, c in clockwise[: len(clockwise) // 2]:
            out[r, c] = seed_color

    return out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, count)
    patches = []
    for color in colors:
        patch = _build_patch(color)
        if patch is None:
            return None
        patches.append(patch)
    return _pack_patches(patches)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
