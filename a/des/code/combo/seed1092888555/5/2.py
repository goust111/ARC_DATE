import random
from collections import deque

import numpy as np


BG_COLOR = 5
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (20, 32)
GEN_RETRIES = 300
PLACE_RETRIES = 250


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                cr, cc = q.popleft()
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            keep = True
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    keep = False
                    break
            out[r, c] = keep
    return out


def _outline_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    eroded = mask.copy()
    for _ in range(thickness):
        eroded = _erode4(eroded)
        if not eroded.any():
            break
    return mask & ~eroded


def _legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}
    if w < 2:
        return mapping, legend_mask
    for r in range(h):
        left = int(grid[r, 0])
        right = int(grid[r, 1])
        if left != bg and right != bg:
            mapping[left] = right
            legend_mask[r, 0] = True
            legend_mask[r, 1] = True
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redraw each object as an inward outline whose color comes from the left-edge
    legend and whose thickness is hole_count + 1.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping, legend_mask = _legend_mapping(g, bg)

    out = np.full_like(g, bg)
    for color, outline_color in mapping.items():
        mask = (g == int(color)) & ~legend_mask
        for comp in _components(mask):
            r0, c0, r1, c1 = _bbox(comp)
            sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                sub[r - r0, c - c0] = True
            thickness = _hole_count(sub) + 1
            outline = _outline_mask(sub, thickness)
            rr, cc = np.nonzero(outline)
            out[r0 + rr, c0 + cc] = int(outline_color)
    return out


def _solid_shape(rng: random.Random) -> np.ndarray:
    style = rng.choice(["rect", "plus", "step"])
    if style == "rect":
        h = rng.randint(4, 8)
        w = rng.randint(4, 8)
        return np.ones((h, w), dtype=bool)

    if style == "plus":
        h = rng.randint(5, 9)
        w = rng.randint(5, 9)
        mask = np.zeros((h, w), dtype=bool)
        vr = rng.randint(2, 3)
        hr = rng.randint(2, 3)
        c0 = rng.randint(1, max(1, w - vr - 1))
        r0 = rng.randint(1, max(1, h - hr - 1))
        mask[:, c0 : c0 + vr] = True
        mask[r0 : r0 + hr, :] = True
        return mask

    h1 = rng.randint(3, 6)
    w1 = rng.randint(4, 7)
    h2 = rng.randint(3, 6)
    w2 = rng.randint(3, 6)
    off_r = rng.randint(1, max(1, h1 - 1))
    off_c = rng.randint(1, max(1, w1 - 1))
    h = max(h1, off_r + h2)
    w = max(w1, off_c + w2)
    mask = np.zeros((h, w), dtype=bool)
    mask[:h1, :w1] = True
    mask[off_r : off_r + h2, off_c : off_c + w2] = True
    return mask


def _holed_shape(rng: random.Random, holes: int) -> np.ndarray:
    orientation = rng.choice(["h", "v"])
    wall = holes + rng.randint(1, 2)
    hole_h = rng.randint(2, 3)
    hole_w = rng.randint(2, 3)
    gap = wall + rng.randint(1, 2)
    pad_h = rng.randint(0, 2)
    pad_w = rng.randint(0, 2)

    if holes == 1:
        outer_h = hole_h + 2 * wall + pad_h
        outer_w = hole_w + 2 * wall + pad_w
        mask = np.ones((outer_h, outer_w), dtype=bool)
        top = wall + rng.randint(0, pad_h)
        left = wall + rng.randint(0, pad_w)
        mask[top : top + hole_h, left : left + hole_w] = False
        return mask

    if orientation == "h":
        outer_h = hole_h + 2 * wall + pad_h
        outer_w = holes * hole_w + (holes - 1) * gap + 2 * wall + pad_w
        mask = np.ones((outer_h, outer_w), dtype=bool)
        top = wall + rng.randint(0, pad_h)
        left = wall + rng.randint(0, pad_w)
        for i in range(holes):
            c0 = left + i * (hole_w + gap)
            mask[top : top + hole_h, c0 : c0 + hole_w] = False
        return mask

    outer_h = holes * hole_h + (holes - 1) * gap + 2 * wall + pad_h
    outer_w = hole_w + 2 * wall + pad_w
    mask = np.ones((outer_h, outer_w), dtype=bool)
    top = wall + rng.randint(0, pad_h)
    left = wall + rng.randint(0, pad_w)
    for i in range(holes):
        r0 = top + i * (hole_h + gap)
        mask[r0 : r0 + hole_h, left : left + hole_w] = False
    return mask


def _build_object(rng: random.Random, holes: int) -> np.ndarray:
    if holes == 0:
        return _solid_shape(rng)
    return _holed_shape(rng, holes)


def _dilate1(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    rr, cc = np.nonzero(mask)
    for r, c in zip(rr, cc):
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _can_place(occupied: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    h, w = shape.shape
    region = occupied[top : top + h, left : left + w]
    if np.any(region & _dilate1(shape)):
        return False
    return True


def _place_shape(grid: np.ndarray, occupied: np.ndarray, shape: np.ndarray, color: int, top: int, left: int):
    h, w = shape.shape
    region = grid[top : top + h, left : left + w]
    region[shape] = color
    occupied[top : top + h, left : left + w] |= shape


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    gh = rng.randint(*GRID_H_RANGE)
    gw = rng.randint(*GRID_W_RANGE)
    grid = np.full((gh, gw), BG_COLOR, dtype=int)
    occupied = np.zeros((gh, gw), dtype=bool)

    n_entries = rng.randint(2, 4)
    palette = [c for c in range(10) if c != BG_COLOR]
    rng.shuffle(palette)
    source_colors = palette[:n_entries]
    outline_colors = palette[n_entries : 2 * n_entries]

    step = rng.choice([2, 3])
    max_span = 1 + (n_entries - 1) * step
    if max_span > gh - 2:
        return None
    start_row = rng.randint(1, gh - max_span - 1)
    legend_rows = [start_row + i * step for i in range(n_entries)]
    for r, src, dst in zip(legend_rows, source_colors, outline_colors):
        grid[r, 0] = src
        grid[r, 1] = dst
        occupied[r, 0] = True
        occupied[r, 1] = True

    hole_counts = [0, 1, 2]
    n_objects = rng.randint(max(3, n_entries), n_entries + 3)
    while len(hole_counts) < n_objects:
        hole_counts.append(rng.choice([0, 1, 2]))
    rng.shuffle(hole_counts)

    object_colors = []
    for color in source_colors:
        object_colors.append(color)
    while len(object_colors) < n_objects:
        object_colors.append(rng.choice(source_colors))
    rng.shuffle(object_colors)

    shapes = [_build_object(rng, hcount) for hcount in hole_counts]
    order = list(range(n_objects))
    rng.shuffle(order)

    min_col = 4
    for idx in order:
        shape = shapes[idx]
        sh, sw = shape.shape
        if sh > gh - 2 or sw > gw - min_col - 1:
            return None
        placed = False
        for _ in range(PLACE_RETRIES):
            top = rng.randint(1, gh - sh - 1)
            left = rng.randint(min_col, gw - sw - 1)
            if not _can_place(occupied, shape, top, left):
                continue
            _place_shape(grid, occupied, shape, object_colors[idx], top, left)
            placed = True
            break
        if not placed:
            return None

    if np.count_nonzero(grid == BG_COLOR) <= (gh * gw) // 2:
        return None
    if not np.array_equal(main(grid), main(np.array(grid, copy=True))):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
