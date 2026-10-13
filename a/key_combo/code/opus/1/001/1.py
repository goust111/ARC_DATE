import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
FILL = 8


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _norm(coords):
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete matched frame holes with right-side fill pieces, then crop the only horizontally symmetric completed rectangle.

    Input structure:
    - A black grid is split by one full grey vertical separator column.
    - Left of the separator are several colored solid rectangles with black connected holes.
    - Right of the separator are cyan connected pieces whose normalized masks match the holes one-to-one.

    Rule:
    - Normalize every left-frame hole and every right-side cyan piece and pair equal masks uniquely.
    - Fill each paired frame hole with cyan, producing complete colored rectangles.
    - Test the cropped completed rectangles for left-right mirror symmetry.
    - Return the unique symmetric completed rectangle crop.

    Key insight:
    - The symmetry test applies only after complementary shape matching has filled the missing hole.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_cols = [c for c in range(w) if np.all(g[:, c] == SEP)]
    assert len(sep_cols) == 1, "expected one grey separator column"
    sep = sep_cols[0]

    right_mask = g[:, sep + 1 :] == FILL
    pieces = {_norm(comp): comp for comp in _components(right_mask)}
    assert pieces, "expected cyan fill pieces"

    completed = []
    left = g[:, :sep]
    for comp in _components((left != BG) & (left != SEP)):
        color = int(left[comp[0]])
        assert color != FILL
        r0, c0, r1, c1 = _bbox(comp)
        sub = left[r0 : r1 + 1, c0 : c1 + 1]
        hole_coords = [(r, c) for r, c in zip(*np.where(sub == BG))]
        assert hole_coords, "frame must contain a hole"
        key = _norm(hole_coords)
        assert key in pieces, "each frame hole must have a matching cyan piece"
        rect = np.full(sub.shape, color, dtype=int)
        for rr, cc in hole_coords:
            rect[rr, cc] = FILL
        completed.append(rect)

    winners = [rect for rect in completed if np.array_equal(rect, np.fliplr(rect))]
    assert len(winners) == 1, "expected one symmetric completed rectangle"
    return winners[0].copy()


def _connected(mask):
    return len(_components(mask)) == 1


def _mirror_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    center = w // 2
    length = random.randint(3, min(5, h - 2))
    r0 = random.randint(1, h - length - 1)
    mask[r0 : r0 + length, center] = True
    if random.random() < 0.45 and center - 1 >= 1 and center + 1 < w - 1:
        rr = r0 + random.randrange(length)
        mask[rr, center - 1] = True
        mask[rr, center + 1] = True
    return mask


def _random_hole(h: int, w: int, want_symmetric: bool, used: set[tuple[tuple[int, int], ...]]) -> np.ndarray:
    for _ in range(200):
        if want_symmetric and w % 2 == 1:
            mask = _mirror_mask(h, w)
        else:
            mask = np.zeros((h, w), dtype=bool)
            r, c = random.randint(1, h - 2), random.randint(1, w - 2)
            mask[r, c] = True
            target = random.randint(3, min(6, (h - 2) * (w - 2)))
            frontier = [(r, c)]
            while int(mask.sum()) < target:
                r, c = random.choice(frontier)
                nbrs = [
                    (r + dr, c + dc)
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 1 <= r + dr < h - 1 and 1 <= c + dc < w - 1 and not mask[r + dr, c + dc]
                ]
                if not nbrs:
                    continue
                nr, nc = random.choice(nbrs)
                mask[nr, nc] = True
                frontier.append((nr, nc))
        key = _norm(list(zip(*np.where(mask))))
        if key in used:
            continue
        if _connected(mask) and (np.array_equal(mask, np.fliplr(mask)) == want_symmetric):
            used.add(key)
            return mask
    raise RuntimeError("failed to sample hole")


def _can_place(grid, patch_mask, r0, c0, gap=1):
    h, w = patch_mask.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(grid.shape[0], r0 + h + gap)
    cc1 = min(grid.shape[1], c0 + w + gap)
    return not np.any(grid[rr0:rr1, cc0:cc1] != BG)


def _place_piece(grid, mask):
    h, w = mask.shape
    for _ in range(200):
        r0 = random.randint(1, grid.shape[0] - h - 1)
        c0 = random.randint(0, grid.shape[1] - w)
        if _can_place(grid, mask, r0, c0):
            grid[r0 : r0 + h, c0 : c0 + w][mask] = FILL
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 4)
    rects = []
    used = set()
    sym_i = random.randrange(n)
    for i in range(n):
        rh = random.randint(5, 6)
        rw = random.choice([5, 7]) if i == sym_i else random.randint(5, 7)
        hole = _random_hole(rh, rw, i == sym_i, used)
        rects.append((rh, rw, hole))

    height = max(18, sum(rh for rh, _, _ in rects) + n + 2)
    width = random.randint(24, 29)
    sep = random.randint(11, 13)
    if sep + 7 >= width:
        return None
    grid = np.zeros((height, width), dtype=int)
    grid[:, sep] = SEP

    row = 1
    colors = random.sample([1, 2, 3, 4], n)
    pieces = []
    for color, (rh, rw, hole) in zip(colors, rects):
        if row + rh >= height or rw + 1 >= sep:
            return None
        c0 = random.randint(0, sep - rw - 1)
        grid[row : row + rh, c0 : c0 + rw] = color
        grid[row : row + rh, c0 : c0 + rw][hole] = BG
        pieces.append(hole.copy())
        row += rh + random.randint(1, 2)

    random.shuffle(pieces)
    right = grid[:, sep + 1 :]
    for mask in pieces:
        rr, cc = np.where(mask)
        r0, c0, r1, c1 = int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())
        piece = mask[r0 : r1 + 1, c0 : c1 + 1]
        if not _place_piece(right, piece):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not np.array_equal(out, main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
