from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
PIECE_COLORS = (1, 2, 4)
MIX = {
    frozenset((1, 1)): 1,
    frozenset((2, 2)): 2,
    frozenset((4, 4)): 4,
    frozenset((1, 2)): 6,
    frozenset((2, 4)): 7,
    frozenset((1, 4)): 3,
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr in range(cr - 1, cr + 2):
                    for nc in range(cc - 1, cc + 2):
                        if nr == cr and nc == cc:
                            continue
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _piece_records(grid: np.ndarray) -> list[dict]:
    records: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for cells in _components(grid == color):
            r0, c0, r1, c1 = _bbox(cells)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
            records.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "shape": mask,
                    "size": mask.shape,
                }
            )
    return records


def _mix_color(a: int, b: int) -> int:
    key = frozenset((int(a), int(b)))
    assert key in MIX, f"unsupported color mix: {a}, {b}"
    return MIX[key]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary monochrome fragments and replace each footprint by the mixed-color rectangle.

    The background is azure.  Each foreground component is a single-color piece whose bounding
    box is the rectangle it came from.  Two pieces pair when their normalized masks are exact
    complements inside equal-size bounding boxes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    records = _piece_records(g)
    assert records and len(records) % 2 == 0

    out = g.copy()
    unused = set(range(len(records)))
    while unused:
        i = min(unused)
        rec = records[i]
        candidates = []
        for j in unused:
            if j == i:
                continue
            other = records[j]
            if rec["size"] != other["size"]:
                continue
            if np.array_equal(rec["shape"], np.logical_not(other["shape"])):
                candidates.append(j)
        assert len(candidates) == 1, f"ambiguous or missing complement for piece {i}: {candidates}"
        j = candidates[0]
        other = records[j]
        color = _mix_color(rec["color"], other["color"])
        for item in (rec, other):
            r0, c0, r1, c1 = item["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = color
        unused.remove(i)
        unused.remove(j)
    return out


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _full_bbox(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _random_complement_pair(h: int, w: int) -> np.ndarray | None:
    for _ in range(2000):
        mask = np.random.random((h, w)) < random.uniform(0.35, 0.65)
        if mask.all() or not mask.any():
            continue
        inv = np.logical_not(mask)
        if not (_connected(mask) and _connected(inv)):
            continue
        if not (_full_bbox(mask) and _full_bbox(inv)):
            continue
        return mask
    return None


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_rectangles(
    canvas_shape: tuple[int, int],
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    for idx in order:
        rh, rw = sizes[idx]
        ok = False
        for _ in range(600):
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if any(prev is not None and _rectangles_overlap(rect, prev, pad=1) for prev in placed):
                continue
            placed[idx] = rect
            ok = True
            break
        if not ok:
            return None
    return [(r0, c0) for r0, c0, _r1, _c1 in placed if r0 is not None]


def _generate_case() -> np.ndarray | None:
    n_pairs = random.randint(2, 5)
    dims_pool = [(h, w) for h in range(4, 9) for w in range(4, 9)]
    random.shuffle(dims_pool)
    dims = dims_pool[:n_pairs]

    masks: list[np.ndarray] = []
    colors: list[tuple[int, int]] = []
    sizes: list[tuple[int, int]] = []
    for h, w in dims:
        mask = _random_complement_pair(h, w)
        if mask is None:
            return None
        masks.append(mask)
        colors.append((random.choice(PIECE_COLORS), random.choice(PIECE_COLORS)))
        sizes.extend([(h, w), (h, w)])

    area = sum((h + 2) * (w + 2) for h, w in sizes)
    side = max(18, int(area**0.5 * 1.8))
    canvas_shape = (random.randint(side, side + 8), random.randint(side, side + 8))
    positions = _place_rectangles(canvas_shape, sizes)
    if positions is None:
        return None

    grid = np.full(canvas_shape, BG, dtype=int)
    pos_idx = 0
    for mask, (color_a, color_b) in zip(masks, colors):
        for piece_mask, color in ((mask, color_a), (np.logical_not(mask), color_b)):
            r0, c0 = positions[pos_idx]
            pos_idx += 1
            rr, cc = np.where(piece_mask)
            grid[r0 + rr, c0 + cc] = int(color)

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment input")
