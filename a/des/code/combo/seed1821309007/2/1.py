import hashlib
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _normalize_cells(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    h = max(rs) - r0 + 1
    w = max(cs) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return _normalize_cells(cells)


def _rotations(mask: np.ndarray):
    variants = []
    seen = set()
    for k in range(4):
        rot = _normalize_mask(np.rot90(mask, k))
        key = rot.tobytes()
        if key not in seen:
            seen.add(key)
            variants.append(rot)
    return variants


def _same_shape_under_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    for rot in _rotations(a):
        if rot.shape == b.shape and np.array_equal(rot, b):
            return True
    return False


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _centered_square_bbox(cells, side: int):
    r0, c0, r1, c1 = _bbox(cells)
    top = (r0 + r1 - (side - 1)) // 2
    left = (c0 + c1 - (side - 1)) // 2
    return top, left, top + side - 1, left + side - 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sample_map = {}
    sample_mask = (g != BG) & (g != GREY)
    for cells in _components(sample_mask):
        touches = any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        if not touches:
            continue
        colors = {int(g[r, c]) for r, c in cells}
        if len(colors) != 1:
            continue
        color = colors.pop()
        sample_map[color] = _normalize_cells(cells)

    out = g.copy()
    out[g == GREY] = BG

    for cells in _components(g == GREY):
        obj = _normalize_cells(cells)
        match_color = None
        for color, sample in sample_map.items():
            if _same_shape_under_rotation(obj, sample):
                match_color = color
                break
        if match_color is None:
            continue
        side = len(cells)
        top, left, bottom, right = _centered_square_bbox(cells, side)
        for r in range(max(0, top), min(h - 1, bottom) + 1):
            for c in range(max(0, left), min(w - 1, right) + 1):
                out[r, c] = match_color

    return out


SHAPE_POOL = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _canonical_key(mask: np.ndarray):
    return min(rot.tobytes() for rot in _rotations(mask))


def _shape_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def _random_edge_position(rng: np.random.Generator, side: str, shape: np.ndarray, h: int, w: int):
    sh, sw = shape.shape
    if side == "top":
        return 0, int(rng.integers(1, w - sw - 1))
    if side == "bottom":
        return h - sh, int(rng.integers(1, w - sw - 1))
    if side == "left":
        return int(rng.integers(1, h - sh - 1)), 0
    return int(rng.integers(1, h - sh - 1)), w - sw


def _cells_from_mask(mask: np.ndarray, top: int, left: int):
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _touching(cells, occupied: np.ndarray, margin: int = 1):
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return True
        r0 = max(0, r - margin)
        r1 = min(h - 1, r + margin)
        c0 = max(0, c - margin)
        c1 = min(w - 1, c + margin)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return True
    return False


def _paint_cells(grid: np.ndarray, cells, color: int):
    for r, c in cells:
        grid[r, c] = color


def _reserve_cells(mask: np.ndarray, cells):
    for r, c in cells:
        mask[r, c] = True


def _build_expected(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    shape_keys = {_canonical_key(shape): shape for shape in SHAPE_POOL}
    unique_shapes = list(shape_keys.values())

    for _ in range(400):
        h = int(rng.integers(16, 23))
        w = int(rng.integers(16, 23))
        grid = np.zeros((h, w), dtype=int)
        occupied_input = np.zeros((h, w), dtype=bool)
        occupied_output = np.zeros((h, w), dtype=bool)

        sample_indices = rng.choice(len(unique_shapes), size=4, replace=False)
        sample_shapes = [unique_shapes[int(i)] for i in sample_indices]
        sample_colors = list(rng.choice(PALETTE, size=4, replace=False))
        sides = ["top", "bottom", "left", "right"]
        sample_map = {}

        ok = True
        for side, base_shape, color in zip(sides, sample_shapes, sample_colors):
            placed = False
            rots = _rotations(base_shape)
            rng.shuffle(rots)
            for rot in rots:
                sh, sw = rot.shape
                if side in ("top", "bottom") and w - sw - 1 <= 1:
                    continue
                if side in ("left", "right") and h - sh - 1 <= 1:
                    continue
                for _try in range(30):
                    top, left = _random_edge_position(rng, side, rot, h, w)
                    cells = _cells_from_mask(rot, top, left)
                    if _touching(cells, occupied_input, margin=0):
                        continue
                    _paint_cells(grid, cells, int(color))
                    _reserve_cells(occupied_input, cells)
                    sample_map[int(color)] = _normalize_mask(rot)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        sample_keys = {_canonical_key(v) for v in sample_map.values()}
        unmatched_shapes = [s for s in unique_shapes if _canonical_key(s) not in sample_keys]
        if not unmatched_shapes:
            continue

        branch = int(rng.integers(0, 3))
        if branch == 0:
            n_match = int(rng.integers(2, 5))
            n_unmatch = 0
        elif branch == 1:
            n_match = int(rng.integers(1, 4))
            n_unmatch = int(rng.integers(1, 3))
        else:
            n_match = int(rng.integers(1, 3))
            n_unmatch = int(rng.integers(2, 5))
        specs = [("match", None)] * n_match + [("unmatch", None)] * n_unmatch
        rng.shuffle(specs)

        for kind, _ in specs:
            if kind == "match":
                color = int(rng.choice(sample_colors))
                base = sample_map[color]
            else:
                base = unmatched_shapes[int(rng.integers(len(unmatched_shapes)))]
            rots = _rotations(base)
            rng.shuffle(rots)
            placed = False
            for rot in rots:
                area = _shape_area(rot)
                sh, sw = rot.shape
                for _try in range(60):
                    top = int(rng.integers(2, h - sh - 2))
                    left = int(rng.integers(2, w - sw - 2))
                    cells = _cells_from_mask(rot, top, left)
                    if _touching(cells, occupied_input, margin=1):
                        continue
                    sq_top, sq_left, sq_bottom, sq_right = _centered_square_bbox(cells, area)
                    if sq_top < 1 or sq_left < 1 or sq_bottom >= h - 1 or sq_right >= w - 1:
                        continue
                    square_cells = [(r, c) for r in range(sq_top, sq_bottom + 1) for c in range(sq_left, sq_right + 1)]
                    if _touching(square_cells, occupied_output, margin=1):
                        continue
                    if _touching(square_cells, occupied_input, margin=0):
                        continue
                    _paint_cells(grid, cells, GREY)
                    _reserve_cells(occupied_input, cells)
                    _reserve_cells(occupied_output, square_cells)
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        grey_comps = _components(grid == GREY)
        if len(grey_comps) != len(specs):
            continue

        expected = _build_expected(grid)
        if not np.array_equal(expected, main(grid)):
            continue

        return grid

    raise RuntimeError("generate_input: failed after 400 attempts")
