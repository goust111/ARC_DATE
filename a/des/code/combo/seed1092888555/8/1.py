import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        dirs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _separator_row(g: np.ndarray) -> int:
    rows = [
        r for r in range(g.shape[0])
        if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != BG
    ]
    assert len(rows) == 1
    return int(rows[0])


def _extract_dictionary(top: np.ndarray) -> dict[int, tuple[int, tuple[tuple[int, int], ...]]]:
    mapping: dict[int, tuple[int, tuple[tuple[int, int], ...]]] = {}
    for comp in _components(top != BG, connectivity=8):
        colors = [int(top[r, c]) for r, c in comp]
        unique = sorted(set(colors))
        assert len(unique) == 2

        counts = {color: colors.count(color) for color in unique}
        key_candidates = [color for color, cnt in counts.items() if cnt == 1]
        assert len(key_candidates) == 1
        key_color = int(key_candidates[0])
        shape_color = int(unique[0] if unique[1] == key_color else unique[1])
        assert key_color not in mapping

        key_pos = next((r, c) for r, c in comp if int(top[r, c]) == key_color)
        shape_cells = [(r, c) for r, c in comp if int(top[r, c]) == shape_color]
        assert shape_cells

        kr, kc = key_pos
        offsets = tuple(sorted((int(r - kr), int(c - kc)) for r, c in shape_cells))
        assert (0, 0) not in offsets
        assert any(max(abs(dr), abs(dc)) == 1 for dr, dc in offsets)
        mapping[key_color] = (shape_color, offsets)

    assert mapping
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the upper panel as a dictionary from singleton key colors to adjacent shapes.

    The grid has a black background and one full-width separator row. Each upper-panel
    object contains exactly two colors: a singleton key pixel and a connected group of
    shape pixels in another color. Each lower-panel foreground pixel is a key marker.
    Copy the lower panel, then for every marker paint the matching shape-color pixels
    at the stored offsets relative to that marker. Drop the upper panel and separator.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_row(g)
    mapping = _extract_dictionary(g[:sep])

    lower = g[sep + 1 :].copy()
    out = lower.copy()
    h, w = out.shape
    for r, c in np.argwhere(lower != BG):
        key_color = int(lower[r, c])
        assert key_color in mapping
        shape_color, offsets = mapping[key_color]
        for dr, dc in offsets:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = shape_color
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _grow_shape(area: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.append(nxt)
        cells.add(random.choice(frontier))
    return _normalize(cells)


def _random_shape() -> tuple[set[tuple[int, int]], tuple[int, int]]:
    area = random.randint(1, 6)
    shape = _grow_shape(area)
    if random.random() < 0.5:
        max_r = max(r for r, _ in shape)
        shape = {(max_r - r, c) for r, c in shape}
    if random.random() < 0.5:
        max_c = max(c for _, c in shape)
        shape = {(r, max_c - c) for r, c in shape}
    if random.random() < 0.5:
        shape = {(c, r) for r, c in shape}
    shape = _normalize(shape)

    neighbors = set()
    for r, c in shape:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            pos = (r + dr, c + dc)
            if pos not in shape:
                neighbors.add(pos)
    key = random.choice(sorted(neighbors))
    return shape, key


def _place_top_objects(
    top_h: int,
    width: int,
    specs: list[tuple[int, int, set[tuple[int, int]], tuple[int, int]]],
) -> np.ndarray | None:
    top = np.full((top_h, width), BG, dtype=int)
    occupied = np.zeros((top_h, width), dtype=bool)

    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        key_color, shape_color, shape, key = specs[idx]
        obj_cells = {(key[0], key[1], key_color)} | {(r, c, shape_color) for r, c in shape}
        rs = [r for r, _, _ in obj_cells]
        cs = [c for _, c, _ in obj_cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        obj_h = max_r - min_r + 1
        obj_w = max_c - min_c + 1
        norm = [(r - min_r, c - min_c, color) for r, c, color in obj_cells]

        candidates = []
        for r0 in range(0, top_h - obj_h + 1):
            for c0 in range(0, width - obj_w + 1):
                ok = True
                for rr, cc, _color in norm:
                    ar, ac = r0 + rr, c0 + cc
                    for nr in range(max(0, ar - 1), min(top_h, ar + 2)):
                        for nc in range(max(0, ac - 1), min(width, ac + 2)):
                            if occupied[nr, nc]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        for rr, cc, color in norm:
            ar, ac = r0 + rr, c0 + cc
            top[ar, ac] = color
            occupied[ar, ac] = True
    return top


def _footprint(offsets: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    return {(0, 0)} | set(offsets)


def _place_bottom_markers(
    bottom_h: int,
    width: int,
    dict_specs: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
    marker_count: int,
) -> np.ndarray | None:
    lower = np.full((bottom_h, width), BG, dtype=int)
    blocked = np.zeros((bottom_h, width), dtype=bool)

    keys = list(dict_specs)
    chosen = [random.choice(keys) for _ in range(marker_count)]
    random.shuffle(chosen)

    for key_color in chosen:
        _shape_color, offsets = dict_specs[key_color]
        cells = _footprint(offsets)
        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)

        candidates = []
        for r in range(-min_r, bottom_h - max_r):
            for c in range(-min_c, width - max_c):
                ok = True
                for dr, dc in cells:
                    rr, cc = r + dr, c + dc
                    for nr in range(max(0, rr - 1), min(bottom_h, rr + 2)):
                        for nc in range(max(0, cc - 1), min(width, cc + 2)):
                            if blocked[nr, nc]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append((r, c))
        if not candidates:
            return None

        r, c = random.choice(candidates)
        lower[r, c] = key_color
        for dr, dc in cells:
            rr, cc = r + dr, c + dc
            blocked[rr, cc] = True
    return lower


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(14, 24)
    width = random.randint(14, 24)
    sep = random.randint(4, height - 7)
    top_h = sep
    bottom_h = height - sep - 1

    dict_count = random.randint(2, min(5, len(PALETTE) // 2))
    marker_count = random.randint(2, 8)
    if bottom_h < 4 or top_h < 4:
        return None

    key_colors = random.sample(PALETTE, dict_count)
    top_specs: list[tuple[int, int, set[tuple[int, int]], tuple[int, int]]] = []
    dict_specs: dict[int, tuple[int, tuple[tuple[int, int], ...]]] = {}

    shape_color_pool = [c for c in PALETTE if c not in key_colors]
    for key_color in key_colors:
        shape, key = _random_shape()
        shape_color = random.choice(shape_color_pool)
        offsets = tuple(sorted((r - key[0], c - key[1]) for r, c in shape))
        dict_specs[key_color] = (shape_color, offsets)
        top_specs.append((key_color, shape_color, shape, key))

    top = _place_top_objects(top_h, width, top_specs)
    if top is None:
        return None

    lower = _place_bottom_markers(bottom_h, width, dict_specs, marker_count)
    if lower is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:sep] = top
    grid[sep, :] = SEP
    grid[sep + 1 :] = lower

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != lower.shape:
        return None
    if np.array_equal(out, lower):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
