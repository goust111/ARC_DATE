from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLORS = tuple(range(10))
GEN_RETRIES = 400
FRAME_COUNT_RANGE = (4, 6)
INTERIOR_SIZE_RANGE = (3, 6)
GRID_SIDE_RANGE = (22, 34)


def _get_bg(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            result.append(np.array(coords, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if np.array_equal(patch, expected):
        return r0, c0, r1, c1
    return None


def _transform_patch(patch: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return patch.copy()
    if k == 2:
        return np.fliplr(patch)
    if k == 3:
        assert patch.shape[0] == patch.shape[1], "90-degree rotation needs a square interior"
        return np.rot90(patch, -1)
    if k == 4:
        return np.rot90(patch, 2)
    raise AssertionError(f"unsupported color-count branch: {k}")


def _extract_frames(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in colors:
        comps = _components(g == color)
        if len(comps) < 2:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for comp in comps:
            box = _frame_bbox(comp)
            if box is None:
                ok = False
                break
            boxes.append(box)
        if ok:
            candidates.append((int(color), boxes))
    assert len(candidates) == 1, "expected a unique color whose components are all rectangular frames"
    frame_color, boxes = candidates[0]
    boxes.sort()
    return frame_color, boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each framed interior according to the number of non-background colors it contains.

    k=1 -> unchanged
    k=2 -> horizontal flip
    k=3 -> rotate 90 degrees clockwise
    k=4 -> rotate 180 degrees
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, frames = _extract_frames(g, bg)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        colors = {int(v) for v in np.unique(interior) if int(v) != bg}
        assert frame_color not in colors, "frame color should not appear inside a frame"
        k = len(colors)
        assert 1 <= k <= 4, "interior color count must be between 1 and 4"
        out[r0 + 1 : r1, c0 + 1 : c1] = _transform_patch(interior, k)

    return out


def _seed_frontier(frontier: set[tuple[int, int]], r: int, c: int, size: int, patch: np.ndarray) -> None:
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size and patch[nr, nc] < 0:
            frontier.add((int(nr), int(nc)))


def _make_pattern(size: int, k: int, bg: int, frame_color: int) -> np.ndarray:
    usable = [c for c in COLORS if c not in {bg, frame_color}]
    cells = [(r, c) for r in range(size) for c in range(size)]
    for _ in range(200):
        colors = random.sample(usable, k)
        patch = np.full((size, size), -1, dtype=int)
        seeds = random.sample(cells, k)
        frontiers: list[set[tuple[int, int]]] = [set() for _ in range(k)]
        colored = 0

        for idx, ((r, c), color) in enumerate(zip(seeds, colors, strict=False)):
            patch[r, c] = int(color)
            colored += 1
            _seed_frontier(frontiers[idx], r, c, size, patch)

        max_cells = min(size * size - 1, max(k + 2, int(round(size * size * random.uniform(0.35, 0.60)))))
        target = random.randint(k, max_cells)

        while colored < target:
            order = list(range(k))
            random.shuffle(order)
            placed = False
            for idx in order:
                options = [cell for cell in frontiers[idx] if patch[cell] < 0]
                if not options:
                    continue
                r, c = random.choice(options)
                patch[r, c] = int(colors[idx])
                colored += 1
                _seed_frontier(frontiers[idx], r, c, size, patch)
                placed = True
                break
            if placed:
                continue
            empties = [(r, c) for r in range(size) for c in range(size) if patch[r, c] < 0]
            if not empties:
                break
            idx = random.randrange(k)
            r, c = random.choice(empties)
            patch[r, c] = int(colors[idx])
            colored += 1
            _seed_frontier(frontiers[idx], r, c, size, patch)

        patch[patch < 0] = bg
        seen_colors = {int(v) for v in np.unique(patch) if int(v) != bg}
        if len(seen_colors) != k:
            continue
        if k > 1 and np.array_equal(patch, _transform_patch(patch, k)):
            continue
        return patch
    raise RuntimeError("failed to sample a non-degenerate interior pattern")


def _place_frames(rows: int, cols: int, sizes: list[int]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)

    for idx in order:
        size = sizes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, rows - size):
            for c0 in range(1, cols - size):
                if occupied[r0 - 1 : r0 + size + 1, c0 - 1 : c0 + size + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[r0 - 1 : r0 + size + 1, c0 - 1 : c0 + size + 1] = True
        placements[idx] = (r0, c0)

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*FRAME_COUNT_RANGE)
    ks = [1, 2, 3, 4] + [random.randint(1, 4) for _ in range(count - 4)]
    random.shuffle(ks)

    bg, frame_color = random.sample(COLORS, 2)
    interior_sizes = [random.randint(*INTERIOR_SIZE_RANGE) for _ in range(count)]
    outer_sizes = [n + 2 for n in interior_sizes]

    total_area = sum(size * size for size in outer_sizes)
    max_size = max(outer_sizes)
    base = max(
        GRID_SIDE_RANGE[0],
        max_size + 4,
        int(np.ceil(np.sqrt(total_area * random.uniform(1.9, 2.4)))),
    )
    if base > GRID_SIDE_RANGE[1]:
        return None

    rows = random.randint(base, GRID_SIDE_RANGE[1])
    cols = random.randint(base, GRID_SIDE_RANGE[1])
    placements = _place_frames(rows, cols, outer_sizes)
    if placements is None:
        return None

    inp = np.full((rows, cols), bg, dtype=int)
    expected = np.full((rows, cols), bg, dtype=int)
    for (r0, c0), interior_size, outer_size, k in zip(placements, interior_sizes, outer_sizes, ks, strict=False):
        patch = _make_pattern(interior_size, k, bg, frame_color)
        _draw_frame(inp, r0, c0, outer_size, frame_color)
        _draw_frame(expected, r0, c0, outer_size, frame_color)
        inp[r0 + 1 : r0 + 1 + interior_size, c0 + 1 : c0 + 1 + interior_size] = patch
        expected[r0 + 1 : r0 + 1 + interior_size, c0 + 1 : c0 + 1 + interior_size] = _transform_patch(patch, k)

    if np.count_nonzero(inp == bg) * 2 <= inp.size:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
