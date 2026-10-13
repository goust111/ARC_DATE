from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

BASE_SHAPES = [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
]


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    shifted = sorted((r - min_r, c - min_c) for r, c in coords)
    return tuple(shifted)


def _shape_variants(shape: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    variants = set()
    current = [(int(r), int(c)) for r, c in shape]
    for _ in range(4):
        current = [(-c, r) for r, c in current]
        variants.add(_normalize(current))
        reflected = [(r, -c) for r, c in current]
        variants.add(_normalize(reflected))
    return sorted(variants)


ALL_SHAPES = []
for _shape in BASE_SHAPES:
    for _variant in _shape_variants(_shape):
        if _variant not in ALL_SHAPES:
            ALL_SHAPES.append(_variant)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _border_side(h: int, w: int, r: int, c: int) -> str:
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    raise AssertionError("marker must lie on border")


def _seed_map(g: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = g.shape
    border_entries: dict[int, list[tuple[int, int, str]]] = {}
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(g[r, c])
            if color == BG:
                continue
            border_entries.setdefault(color, []).append((r, c, _border_side(h, w, r, c)))

    assert border_entries, "expected colored border markers"
    seeds: dict[int, tuple[int, int]] = {}
    for color, marks in border_entries.items():
        assert len(marks) == 2, "each marker color must appear exactly twice"
        vertical_marks = [(r, c, side) for r, c, side in marks if side in {"top", "bottom"}]
        horizontal_marks = [(r, c, side) for r, c, side in marks if side in {"left", "right"}]
        assert len(vertical_marks) == 1 and len(horizontal_marks) == 1, "each color must mark one row side and one column side"
        seed = (int(horizontal_marks[0][0]), int(vertical_marks[0][1]))
        sr, sc = seed
        assert 1 <= sr < h - 1 and 1 <= sc < w - 1, "seed must be inside the border"
        seeds[int(color)] = seed
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer hidden seed points from paired border markers, then recolor each grey object
    to the color of the nearest seed to that object's visual center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 7 and w >= 7, "grid must leave room for border markers and interior objects"

    seeds = _seed_map(g)
    comps = _connected_components(g == GREY)
    assert comps, "expected grey interior objects"

    out = g.copy()
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        assert all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in comp), "grey objects must stay inside the border"
        center = (_round_half_up(sum(rows) / len(rows)), _round_half_up(sum(cols) / len(cols)))
        scored = []
        for color, seed in seeds.items():
            dr = center[0] - seed[0]
            dc = center[1] - seed[1]
            scored.append((dr * dr + dc * dc, color))
        scored.sort()
        assert len(scored) == 1 or scored[0][0] < scored[1][0], "nearest seed must be unique"
        color = int(scored[0][1])
        for r, c in comp:
            out[r, c] = color
    return out


def _valid_seed_layout(h: int, w: int, seeds: list[tuple[int, int]]) -> bool:
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a = seeds[i]
            b = seeds[j]
            if a[0] == b[0] or a[1] == b[1]:
                return False
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 5:
                return False
    return True


def _sample_seeds(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    rows = list(range(2, h - 2))
    cols = list(range(2, w - 2))
    if len(rows) < count or len(cols) < count:
        return None
    for _ in range(200):
        sample_rows = random.sample(rows, count)
        sample_cols = random.sample(cols, count)
        random.shuffle(sample_cols)
        seeds = list(zip(sample_rows, sample_cols))
        if _valid_seed_layout(h, w, seeds):
            return [(int(r), int(c)) for r, c in seeds]
    return None


def _paint_markers(grid: np.ndarray, seeds_by_color: dict[int, tuple[int, int]]) -> None:
    h, w = grid.shape
    for color, (r, c) in seeds_by_color.items():
        grid[0 if random.random() < 0.5 else h - 1, c] = color
        grid[r, 0 if random.random() < 0.5 else w - 1] = color


def _touches_existing(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if grid[r, c] != BG:
            return True
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in cell_set:
                continue
            if grid[nr, nc] == GREY:
                return True
    return False


def _object_center(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (_round_half_up(sum(rows) / len(rows)), _round_half_up(sum(cols) / len(cols)))


def _nearest_seed(center: tuple[int, int], seeds_by_color: dict[int, tuple[int, int]]) -> tuple[int, int] | None:
    scored = []
    for color, seed in seeds_by_color.items():
        dr = center[0] - seed[0]
        dc = center[1] - seed[1]
        scored.append((dr * dr + dc * dc, color))
    scored.sort()
    if len(scored) > 1 and scored[1][0] - scored[0][0] < 4:
        return None
    return int(scored[0][1]), int(scored[0][0])


def _candidate_positions(
    h: int,
    w: int,
    shape: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    return [(r0, c0) for r0 in range(1, h - 1 - shape_h + 1) for c0 in range(1, w - 1 - shape_w + 1)]


def _place_object_for_seed(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    target_color: int,
    seeds_by_color: dict[int, tuple[int, int]],
) -> bool:
    h, w = grid.shape
    starts = _candidate_positions(h, w, shape)
    random.shuffle(starts)
    for r0, c0 in starts:
        cells = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if any(cell in seeds_by_color.values() for cell in cells):
            continue
        if _touches_existing(grid, cells):
            continue
        center = _object_center(cells)
        nearest = _nearest_seed(center, seeds_by_color)
        if nearest is None or nearest[0] != target_color:
            continue
        for r, c in cells:
            grid[r, c] = GREY
        return True
    return False


def _generate_case() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    seed_count = random.randint(3, 4)
    seeds = _sample_seeds(h, w, seed_count)
    if seeds is None:
        return None

    colors = random.sample(PALETTE, seed_count)
    seeds_by_color = {color: seed for color, seed in zip(colors, seeds)}

    grid = np.full((h, w), BG, dtype=int)
    _paint_markers(grid, seeds_by_color)

    object_targets = colors[:]
    extra_count = random.randint(1, 5)
    object_targets.extend(random.choice(colors) for _ in range(extra_count))
    random.shuffle(object_targets)

    for target_color in object_targets:
        placed = False
        shapes = ALL_SHAPES[:]
        random.shuffle(shapes)
        for shape in shapes:
            if _place_object_for_seed(grid, shape, target_color, seeds_by_color):
                placed = True
                break
        if not placed:
            return None

    grey_count = int(np.sum(grid == GREY))
    if grey_count < 9:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _generate_case()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
