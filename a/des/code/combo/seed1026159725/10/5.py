import random
from collections import Counter, deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SHAPE_TYPES = ("H", "V", "L")
L_ORIENTATIONS = ("dr", "dl", "ur", "ul")

GRID_SIDE_RANGE = (12, 18)
OBJECT_COUNT_RANGE = (3, 5)
BORDER_PAD = 1
LINE_LENGTH_RANGE = (2, 7)
L_ARM_RANGE = (2, 5)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 250
SEQUENCE_RETRIES = 100


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append(coords)
    return components


def _path_neighbors(coords: list[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    coord_set = set(coords)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        adjacent = []
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in coord_set:
                adjacent.append(nxt)
        neighbors[(r, c)] = adjacent
    return neighbors


def order_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    neighbors = _path_neighbors(coords)
    degrees = {cell: len(adj) for cell, adj in neighbors.items()}
    assert max(degrees.values(), default=0) <= 2, "foreground object must be a thin path"
    if len(coords) == 1:
        return [coords[0]]

    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path object must have exactly two endpoints"

    start = min(endpoints)
    ordered: list[tuple[int, int]] = []
    prev = None
    cur = start
    while True:
        ordered.append(cur)
        candidates = [nxt for nxt in neighbors[cur] if nxt != prev]
        if not candidates:
            break
        assert len(candidates) == 1, "path traversal must not branch"
        prev, cur = cur, candidates[0]

    assert len(ordered) == len(coords), "path traversal must visit every pixel exactly once"
    return ordered


def _palindromize(seq: list[int]) -> list[int]:
    result = list(seq)
    for i in range(len(result) // 2):
        result[-1 - i] = result[i]
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """Rewrite each path object's color sequence into a palindrome along the path."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = get_bg(g)
    components = find_components(g, bg)
    assert components, "expected at least one foreground object"

    out = g.copy()
    for comp in components:
        path = order_path(comp)
        colors = [int(g[r, c]) for r, c in path]
        pal = _palindromize(colors)
        for (r, c), color in zip(path, pal):
            out[r, c] = int(color)
    return out


def _expand_forbidden(coords: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked.add((nr, nc))
    return blocked


def _line_lengths(limit: int, parity: int) -> list[int]:
    lo, hi = LINE_LENGTH_RANGE
    return [length for length in range(lo, min(hi, limit) + 1) if length % 2 == parity]


def _l_arm_pairs(max_h: int, max_w: int, parity: int) -> list[tuple[int, int]]:
    pairs = []
    for arm_r in range(L_ARM_RANGE[0], min(L_ARM_RANGE[1], max_h) + 1):
        for arm_c in range(L_ARM_RANGE[0], min(L_ARM_RANGE[1], max_w) + 1):
            if (arm_r + arm_c - 1) % 2 == parity:
                pairs.append((arm_r, arm_c))
    return pairs


def _build_l_shape(arm_r: int, arm_c: int, orientation: str) -> list[tuple[int, int]]:
    rel = {(0, j) for j in range(arm_c)} | {(i, 0) for i in range(arm_r)}
    transformed = []
    for r, c in rel:
        if orientation == "dr":
            rr, cc = r, c
        elif orientation == "dl":
            rr, cc = r, -c
        elif orientation == "ur":
            rr, cc = -r, c
        else:
            rr, cc = -r, -c
        transformed.append((rr, cc))

    min_r = min(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    return [(r - min_r, c - min_c) for r, c in transformed]


def _place_shape(
    shape_type: str,
    parity: int,
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    row_lo = BORDER_PAD
    row_hi = h - BORDER_PAD - 1
    col_lo = BORDER_PAD
    col_hi = w - BORDER_PAD - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PLACEMENT_RETRIES):
        if shape_type == "H":
            lengths = _line_lengths(col_hi - col_lo + 1, parity)
            if not lengths:
                return None
            length = random.choice(lengths)
            r = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi - length + 1)
            coords = [(r, c0 + dc) for dc in range(length)]
        elif shape_type == "V":
            lengths = _line_lengths(row_hi - row_lo + 1, parity)
            if not lengths:
                return None
            length = random.choice(lengths)
            r0 = random.randint(row_lo, row_hi - length + 1)
            c = random.randint(col_lo, col_hi)
            coords = [(r0 + dr, c) for dr in range(length)]
        else:
            pairs = _l_arm_pairs(row_hi - row_lo + 1, col_hi - col_lo + 1, parity)
            if not pairs:
                return None
            arm_r, arm_c = random.choice(pairs)
            rel_coords = _build_l_shape(arm_r, arm_c, random.choice(L_ORIENTATIONS))
            max_r = max(r for r, _ in rel_coords)
            max_c = max(c for _, c in rel_coords)
            r0 = random.randint(row_lo, row_hi - max_r)
            c0 = random.randint(col_lo, col_hi - max_c)
            coords = [(r0 + r, c0 + c) for r, c in rel_coords]

        if any(cell in forbidden for cell in coords):
            continue
        return coords
    return None


def _sample_sequence(length: int, bg: int) -> list[int] | None:
    palette = [color for color in range(10) if color != bg]
    for _ in range(SEQUENCE_RETRIES):
        color_count = random.randint(2, min(4, len(palette)))
        colors = random.sample(palette, color_count)
        seq = [random.choice(colors) for _ in range(length)]
        if len(set(seq)) < 2:
            continue
        if seq == seq[::-1]:
            continue
        return seq
    return None


def _shape_schedule(count: int) -> list[str]:
    shapes = list(SHAPE_TYPES)
    random.shuffle(shapes)
    while len(shapes) < count:
        shapes.append(random.choice(SHAPE_TYPES))
    random.shuffle(shapes)
    return shapes


def _parity_schedule(count: int) -> list[int]:
    parities = [0, 1]
    while len(parities) < count:
        parities.append(random.randint(0, 1))
    random.shuffle(parities)
    return parities


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    bg = random.randint(0, 9)
    grid = np.full((h, w), bg, dtype=int)

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    shape_types = _shape_schedule(object_count)
    parities = _parity_schedule(object_count)

    forbidden: set[tuple[int, int]] = set()
    all_paths: list[list[tuple[int, int]]] = []
    for shape_type, parity in zip(shape_types, parities):
        coords = _place_shape(shape_type, parity, h, w, forbidden)
        if coords is None:
            return None
        path = order_path(coords)
        seq = _sample_sequence(len(path), bg)
        if seq is None:
            return None
        for (r, c), color in zip(path, seq):
            grid[r, c] = color
        forbidden |= _expand_forbidden(coords, h, w)
        all_paths.append(path)

    if len(find_components(grid, bg)) != object_count:
        return None

    for path in all_paths:
        seq = [int(grid[r, c]) for r, c in path]
        if seq == seq[::-1]:
            return None

    transformed = main(grid)
    if np.array_equal(transformed, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palindrome-path ARC input")
