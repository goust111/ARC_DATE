import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 0
LOOP_COLOR = 5
VOTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 30
OBJECT_COUNT_MIN = 1
OBJECT_COUNT_MAX = 4
PLACEMENT_TRIES = 120
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _interior_from_component(coords: np.ndarray) -> list[tuple[int, int]]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())

    barrier = np.zeros((r1 - r0 + 3, c1 - c0 + 3), dtype=bool)
    for r, c in coords:
        barrier[int(r) - r0 + 1, int(c) - c0 + 1] = True

    h, w = barrier.shape
    seen = np.zeros_like(barrier, dtype=bool)
    queue = deque()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if barrier[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue.append((r, c))

    while queue:
        cr, cc = queue.popleft()
        for nr, nc in _neighbors4(cr, cc):
            if 0 <= nr < h and 0 <= nc < w and not barrier[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    interior: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if barrier[r, c] or seen[r, c]:
                continue
            interior.append((r0 + r - 1, c0 + c - 1))
    return interior


def _is_simple_loop(mask: np.ndarray) -> bool:
    components = _components_from_mask(mask)
    if len(components) != 1:
        return False

    coord_set = {tuple(map(int, cell)) for cell in components[0]}
    for r, c in coord_set:
        degree = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))
        if degree != 2:
            return False

    return bool(_interior_from_component(components[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each gray loop to the majority interior vote color and clear its interior.

    Input structure:
    - The background is black.
    - Gray pixels form one or more disjoint 4-connected single-pixel closed loops.
    - Inside each loop, scattered non-black non-gray pixels act as vote pixels and may use two or three colors.

    Rule:
    - Process each gray loop independently.
    - Find the cells strictly enclosed by that loop, count the interior vote pixels by color, and take the unique majority color.
    - Recolor every gray boundary pixel of the loop to that majority color.
    - Clear every enclosed interior cell to black.
    - Leave everything outside all loops unchanged.

    Key insight:
    - The loop boundary does not determine its own output color; the decisive signal is the majority vote among the colored pixels trapped inside the loop.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for component in _components_from_mask(g == LOOP_COLOR):
        interior = _interior_from_component(component)
        assert interior, "each gray loop must enclose a non-empty region"

        votes = [
            int(g[r, c])
            for r, c in interior
            if int(g[r, c]) not in (BG_COLOR, LOOP_COLOR)
        ]
        counts = Counter(votes)
        assert counts, "each enclosed region must contain vote pixels"

        ranked = counts.most_common()
        assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "majority color must be unique"
        majority_color = int(ranked[0][0])

        out[component[:, 0], component[:, 1]] = majority_color
        for r, c in interior:
            out[r, c] = BG_COLOR

    return out


def _path_mask(height: int, width: int, segments: list[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    r = 0
    c = 0
    mask[r, c] = True

    for dr, dc, steps in segments:
        for _ in range(steps):
            r += dr
            c += dc
            mask[r, c] = True

    assert (r, c) == (0, 0)
    return mask


def _rectangle_loop_mask(height: int, width: int) -> np.ndarray:
    return _path_mask(
        height,
        width,
        [
            (0, 1, width - 1),
            (1, 0, height - 1),
            (0, -1, width - 1),
            (-1, 0, height - 1),
        ],
    )


def _notched_loop_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    notch_top = rng.randint(2, height - 5)
    notch_height = rng.randint(2, height - notch_top - 3)
    notch_width = rng.randint(2, width - 3)

    base = _path_mask(
        height,
        width,
        [
            (0, 1, width - 1),
            (1, 0, notch_top),
            (0, -1, notch_width),
            (1, 0, notch_height),
            (0, 1, notch_width),
            (1, 0, height - 1 - notch_top - notch_height),
            (0, -1, width - 1),
            (-1, 0, height - 1),
        ],
    )

    transforms = (
        lambda x: x,
        np.fliplr,
        lambda x: x.T,
        lambda x: np.flipud(x.T),
    )
    return transforms[rng.randrange(len(transforms))](base)


def _positive_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values: list[int] = []
    previous = 0
    for cut in cuts + [total]:
        values.append(cut - previous)
        previous = cut
    return values


def _make_object(rng: random.Random, object_count: int) -> np.ndarray:
    crowded = object_count >= 3
    if rng.random() < 0.5:
        max_side = 8 if crowded else 10
        height = rng.randint(5, max_side)
        width = rng.randint(5, max_side)
        loop_mask = _rectangle_loop_mask(height, width)
    else:
        max_side = 9 if crowded else 11
        height = rng.randint(7, max_side)
        width = rng.randint(7, max_side)
        loop_mask = _notched_loop_mask(height, width, rng)

    assert _is_simple_loop(loop_mask)
    loop_coords = _components_from_mask(loop_mask)[0]
    interior = _interior_from_component(loop_coords)

    n_colors = 2 if rng.random() < 0.5 else 3
    colors = rng.sample(VOTE_COLORS, n_colors)
    majority_color = colors[rng.randrange(n_colors)]
    other_colors = [color for color in colors if color != majority_color]

    min_votes = 2 * n_colors - 1
    max_votes = min(len(interior), max(min_votes, int(round(len(interior) * 0.8))))
    total_votes = rng.randint(min_votes, max_votes)
    majority_count = rng.randint(total_votes // 2 + 1, total_votes - (n_colors - 1))
    other_counts = _positive_partition(total_votes - majority_count, n_colors - 1, rng)

    vote_colors: list[int] = [majority_color] * majority_count
    shuffled_others = other_colors[:]
    rng.shuffle(shuffled_others)
    for color, count in zip(shuffled_others, other_counts):
        vote_colors.extend([color] * count)
    rng.shuffle(vote_colors)

    obj = np.full(loop_mask.shape, BG_COLOR, dtype=int)
    obj[loop_mask] = LOOP_COLOR

    chosen_cells = rng.sample(interior, total_votes)
    for (r, c), color in zip(chosen_cells, vote_colors):
        obj[r, c] = color

    counts = Counter(vote_colors)
    ranked = counts.most_common()
    assert ranked[0][0] == majority_color
    assert ranked[0][1] > ranked[1][1]
    return obj


def _valid_generated_grid(grid: np.ndarray) -> bool:
    components = _components_from_mask(grid == LOOP_COLOR)
    if not (OBJECT_COUNT_MIN <= len(components) <= OBJECT_COUNT_MAX):
        return False

    for component in components:
        rows = component[:, 0]
        cols = component[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())

        local = grid[r0:r1 + 1, c0:c1 + 1] == LOOP_COLOR
        if not _is_simple_loop(local):
            return False

        interior = _interior_from_component(component)
        colors = [
            int(grid[r, c])
            for r, c in interior
            if int(grid[r, c]) not in (BG_COLOR, LOOP_COLOR)
        ]
        if len(set(colors)) not in (2, 3):
            return False

        counts = Counter(colors).most_common()
        if len(counts) < 2 or counts[0][1] <= counts[1][1]:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    min_side = 10 + 2 * object_count
    height = rng.randint(min_side, GRID_SIDE_MAX)
    width = rng.randint(min_side, GRID_SIDE_MAX)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    for _ in range(object_count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            obj = _make_object(rng, object_count)
            oh, ow = obj.shape
            if oh + 2 > height or ow + 2 > width:
                continue

            top = rng.randint(1, height - oh - 1)
            left = rng.randint(1, width - ow - 1)
            rs = max(0, top - 1)
            re = min(height, top + oh + 1)
            cs = max(0, left - 1)
            ce = min(width, left + ow + 1)
            if blocked[rs:re, cs:ce].any():
                continue

            region = obj != BG_COLOR
            target = grid[top:top + oh, left:left + ow]
            if np.any(target[region] != BG_COLOR):
                continue

            target[region] = obj[region]
            blocked[rs:re, cs:ce] = True
            placed = True
            break

        if not placed:
            return None

    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ARC input")
