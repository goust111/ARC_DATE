from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _component_records(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    records = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(g[sr, sc]) == BG:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            colors = set()
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                colors.add(int(g[r, c]))
                for dr, dc in NEIGHBORS8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            assert len(colors) == 1, "each colored component must be monochrome"
            r0, c0, r1, c1 = _bbox_from_points(cells)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
            records.append({"color": colors.pop(), "mask": mask, "bbox": (r0, c0, r1, c1)})

    return records


def _try_pair(a: dict, b: dict) -> np.ndarray | None:
    ma = a["mask"]
    mb0 = b["mask"]
    if ma.shape != mb0.shape:
        return None

    for mb in (mb0, np.rot90(mb0, 2)):
        overlap = ma & mb
        union = ma | mb
        if not bool(overlap.any()) and bool(union.all()):
            rect = np.full(ma.shape, BG, dtype=int)
            rect[ma] = int(a["color"])
            rect[mb] = int(b["color"])
            return rect
    return None


def _find_pairs(records: list[dict]) -> list[np.ndarray]:
    n = len(records)
    assert n > 0 and n % 2 == 0, "there must be an even number of pieces"

    def search(remaining: tuple[int, ...]) -> list[np.ndarray] | None:
        if not remaining:
            return []
        first = remaining[0]
        rest = set(remaining[1:])
        for second in remaining[1:]:
            rect = _try_pair(records[first], records[second])
            if rect is None:
                continue
            next_remaining = tuple(i for i in remaining[1:] if i != second)
            tail = search(next_remaining)
            if tail is not None:
                return [rect] + tail
        return None

    result = search(tuple(range(n)))
    assert result is not None, "pieces must have a complete complementary matching"
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary scattered pieces into rectangles and lay them out by area.

    Input structure:
    - Color 0 is the black background.
    - The non-background cells form an even number of separated 8-connected components.
    - Every component is monochrome and has an irregular silhouette inside its own bounding box.
    - Components can be paired by equal bounding-box size; one piece in a pair may need a 180-degree rotation.

    Rule:
    - Crop every colored component to its minimal bounding box and keep a binary mask of its colored cells.
    - Match components whose masks are exact complements inside the same rectangle, allowing a 180-degree rotation of one mask.
    - Overlay each matched pair into a solid rectangle, preserving the original color of every source pixel.
    - Sort the completed rectangles by increasing area, then place them left to right with one black column between them and bottom edges aligned.

    Key insight:
    - The pairing is encoded by negative space: a piece's black cells inside its crop are exactly the cells occupied by its partner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    records = _component_records(g)
    rectangles = _find_pairs(records)
    rectangles.sort(key=lambda r: (int(r.size), int(r.shape[0]), int(r.shape[1])))

    out_h = max(int(r.shape[0]) for r in rectangles)
    out_w = sum(int(r.shape[1]) for r in rectangles) + len(rectangles) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for rect in rectangles:
        h, w = rect.shape
        out[out_h - h: out_h, c: c + w] = rect
        c += w + 1
    return out


# --------------- generator ---------------

DIM_POOL = [
    (2, 3), (3, 3), (2, 5), (3, 4), (2, 7), (3, 5),
    (4, 4), (4, 5), (3, 7), (4, 6), (5, 5), (5, 6),
]
GEN_RETRIES = 400
MASK_RETRIES = 2000
PLACE_RETRIES = 300


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]):
                p = (int(nr), int(nc))
                if p not in seen:
                    seen.add(p)
                    q.append(p)
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _valid_half(mask: np.ndarray) -> bool:
    area = int(mask.sum())
    total = int(mask.size)
    if area < 2 or total - area < 2:
        return False
    return _touches_all_sides(mask) and _touches_all_sides(~mask) and _is_connected(mask) and _is_connected(~mask)


def _make_complement_mask(h: int, w: int) -> np.ndarray | None:
    total = h * w
    for _ in range(MASK_RETRIES):
        prob = random.uniform(0.35, 0.65)
        mask = np.random.random((h, w)) < prob
        area = int(mask.sum())
        if area < max(2, total // 4) or area > min(total - 2, (total * 3) // 4):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        if _valid_half(mask):
            return mask
    return None


def _choose_dimensions(n_pairs: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        dims = random.sample(DIM_POOL, n_pairs)
        areas = [h * w for h, w in dims]
        if len(set(areas)) != len(areas):
            continue
        if sum(w for _, w in dims) + n_pairs - 1 <= 30 and max(h for h, _ in dims) <= 30:
            return dims
    return None


def _paint_piece(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[r0 + rr, c0 + cc] = int(color)


def _scatter_pieces(pieces: list[tuple[np.ndarray, int]]) -> np.ndarray | None:
    random.shuffle(pieces)
    h = random.randint(17, 30)
    w = random.randint(17, 30)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    pieces.sort(key=lambda p: int(p[0].size), reverse=True)
    for mask, color in pieces:
        ph, pw = mask.shape
        if ph + 2 > h or pw + 2 > w:
            return None
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(1, h - ph - 1)
            c0 = random.randint(1, w - pw - 1)
            if blocked[r0 - 1: r0 + ph + 1, c0 - 1: c0 + pw + 1].any():
                continue
            _paint_piece(grid, mask, color, r0, c0)
            blocked[r0 - 1: r0 + ph + 1, c0 - 1: c0 + pw + 1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(2, 4)
    dims = _choose_dimensions(n_pairs)
    if dims is None:
        return None

    colors = random.sample(range(1, 10), 2 * n_pairs)
    pieces: list[tuple[np.ndarray, int]] = []
    for i, (h, w) in enumerate(dims):
        mask = _make_complement_mask(h, w)
        if mask is None:
            return None

        other = ~mask
        if random.randrange(2) == 1:
            other = np.rot90(other, 2)

        pieces.append((mask.copy(), colors[2 * i]))
        pieces.append((other.copy(), colors[2 * i + 1]))

    grid = _scatter_pieces(pieces)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(_component_records(grid)) != 2 * n_pairs:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-pieces input")
